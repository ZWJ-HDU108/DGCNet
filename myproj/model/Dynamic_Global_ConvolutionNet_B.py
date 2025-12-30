import warnings
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from timm.layers import DropPath, to_2tuple
from torch.utils.checkpoint import checkpoint

try:
    from natten.functional import na2d_av
    has_natten = True
except:
    has_natten = False
    warnings.warn("The efficiency may be reduced since 'natten' is not installed."
                  " It is recommended to install natten for better performance.")

def stem(in_chans, embed_dim):  # [B, C, H, W] -> [B, C, H, W]
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.SiLU(),
    )

def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):

    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


# Choose which normalization to use
def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)  # Synchronization batch normalization
    else:
        return nn.BatchNorm2d(dim)  # 2D batch normalization

def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i: i+1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


# Adaptively emphasize important channels, suppress unimportant channels, and improve the ability of feature representation
class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
            nn.Conv2d(dim, inner_dim, kernel_size=1),  # [B, C, 1, 1] -> [B, inner_dim, 1, 1]
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),  # [B, inner_dim, 1, 1] -> [B, dim, 1, 1]
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)  # [B, C, H, W] -> [B, C, H, W]
        return x


# Channel by channel learnable scaling mechanism
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):

        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])  # [B, C, H, W] -> [B, C, H, W]

        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()



class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))  # Learning parameters as scaling factors
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))  # The learning parameter, as a bias term, is used to translate the normalized feature


    # The process dimension does not change
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)  # Calculate the L2 norm of the input tensor x on the spatial dimensions h, w
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)  # normalization
        # Dynamically adjust the response strength of each input channel
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)  # Main volume integral branch
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        #  Multiscale dilation convolution branching (training mode only)
        if kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy: # Only use the above expansion convolution without deployment
            self.origin_bn = get_bn(channels, use_sync_bn)  # Select normalization, which should be batch normalization by default
            for k, r in zip(self.kernel_sizes, self.dilates):
                # Create inflated convolution
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2,  # Key: padding with unchanged size
                                           dilation=r,  # Expansion rate
                                           groups=channels,  # Deep separable convolution
                                           bias=False))  # [B, C, H, W] -> [B, C, H, W]
                # Create a corresponding BN layer for each dilation convolution
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        # Deployment mode judgment
        if not hasattr(self, 'origin_bn'):  # Check whether the origin_bn attribute exists
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))  # Main branch processing
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))  # Get convolution layer
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))  # Get BN layer
            out = out + bn(conv(x))  # Multi branch fusion
        return out  # [B, C, H, W] -> [B, C, H, W]

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class ResDWConv(nn.Conv2d):
    '''
    Depthwise conv with residual connection
    assume in shape: [B, C, H, W], then out shape-> [B, C, H, W]
    '''
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x


class ContMixBlock(nn.Module):
    '''
    A plug-and-play implementation of ContMix module with FFN layer
    Paper: https://arxiv.org/abs/2502.20087
    '''
    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 res_scale=False,
                 ls_init_value=None,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 **kwargs):

        super().__init__()
        '''
        Args:
        kernel_size: kernel size of the main ContMix branch, default is 7
        smk_size: kernel size of the secondary ContMix branch, default is 5
        num_heads: number of dynamic kernel heads, default is 2
        mlp_ratio: ratio of mlp hidden dim to embedding dim, default is 4
        res_scale: whether to use residual layer scale, default is False
        ls_init_value: layer scale init value, default is None
        drop_path: drop path rate, default is 0
        norm_layer: normalization layer, default is LayerNorm2d
        use_gemm: whether to use iGEMM implementation for large kernel conv, default is False
        deploy: whether to use deploy mode, default is False
        use_checkpoint: whether to use grad checkpointing, default is False
        **kwargs: other arguments
        '''
        mlp_dim = int(dim*mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.dwconv1 = ResDWConv(dim, kernel_size=3)  # Deep separable convolution and residual connection module
        self.norm1 = norm_layer(dim)  # 2D layer normalization

        # 用来计算亲和度矩阵A=========================================
        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7), # first through adaptive pooling
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )
        self.weight_value = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),  # Generate values matrix
        )

        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)  # Projection block, convenient for subsequent attention splitting
        self.fusion_proj = nn.Sequential(  # Fusion block, the number of channels remains unchanged, and only the weight is reshaped
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )

        self.se_layer = SEModule(dim)  # Squeeze and exception enhances the response of the network to important features by adaptively adjusting the channel weights

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        # Normal projection layer
        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        self.dwconv2 = ResDWConv(dim, kernel_size=3)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
        )

        self.ls1 = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.trunc_normal_(self.rpb1, std=0.02)
        nn.init.trunc_normal_(self.rpb2, std=0.02)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1  # Calculate relative position offset
        idx_h = torch.arange(0, kernel_size)  # Generate base index [0, 1, 2,..., k-1]
        idx_w = torch.arange(0, kernel_size)  # [0, 1, 2,..., k-1]
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)  # Idx_h: absolute index in height direction, idx_w: absolute index in width direction, idx_k: one-dimensional index in relative position offset table

    # Apply relative position offset
    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """
        RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3
        """
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)  # Create a full 1 matrix with the same size as the core
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb  # Bring location information to the attention

    def reparm(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()

    def _forward_inner(self, x):
        input_resolution = x.shape[2:]  # from [B, C, H, W] return [H, W]
        B, C, H, W = x.shape  # for our data size [64, 256, 16, 16]

        x = self.dwconv1(x)  # [B, C, H, W] -> [B, C, H, W]

        identity = x

        x = self.norm1(x)  # [B, C, H, W] -> [B, C, H, W]
        gate = self.gate(x)  # [B, C, H, W] -> [B, C, H/2, W/2]

        lepe = self.lepe(x)

        is_pad = False
        if min(H, W) < self.kernel_size:
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)  # upsampling or downsampling of tensors
            H, W = size

        query = self.weight_query(x) * self.scale  # [B, C, H, W] ->[B, C/2, H, W]
        key = self.weight_key(x)  # [B, C, H, W] ->[B, C/2, H, W]
        value = self.weight_value(x)  # [B, C, H, W] ->[B, C, H, W]

        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)  # default num heads=4 [B, g×C, H, W] -> [B, g, C, H×W]
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)  # default num heads=4 [B, g×C, H, W] -> [B, g, C, H×W]
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')  # Matrix multiplication, generating Affinity Matrix A  -> [B, 4, H×W, H×W]
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()  # [B, 4, H×W, H×W] -> [B, H×W, 4, H×W]
        weight = self.weight_proj(weight)  # [B, H×W, 4, H×W] -> [B, kernel_size**2 + smk_size**2, 4, H×W]
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)  # shape: [B, 4, H, W, H×W=kernel_size**2 + smk_size**2]

        # default smk_size = 5, kernel_size = 7     H×W = smk_size² + kernel_size²
        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**2, self.kernel_size**2], dim=-1)
        # attn1: [B, 4, H, W, smk_size^2], attn2: [B, 4, H, W, kernel_size^2]

        # Get position index of convolution================================
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)

        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)  # Attention with relative position information
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        # The above four line dimensions are [B, 4, H, W, smk_size^2] or [B, 4, H, W, kernel_size^2]
        value = rearrange(value, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)  # [B, 2×4×C, H, W] -> [2, B, 4, H, W, C] Here the 0th dimension is 2, which is prepared for the following global convolution operation

        if has_natten:  # This step implements the global perceptual convolution operation
            x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)  # Small nucleus attention+small nucleus value
            x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)  # Large core attention+large core value
        else:
            pad1 = self.smk_size // 2
            pad2 = self.kernel_size // 2
            H_o1 = H - 2 * pad1
            W_o1 = W - 2 * pad1
            H_o2 = H - 2 * pad2
            W_o2 = W - 2 * pad2

            v1 = rearrange(value[0], 'b g h w c -> b (g c) h w')  # [B, g, H, W, C] -> [B, g×C, H, W]
            v2 = rearrange(value[1], 'b g h w c -> b (g c) h w')

            v1 = F.unfold(v1, kernel_size=self.smk_size).reshape(B, -1, H_o1, W_o1)
            v2 = F.unfold(v2, kernel_size=self.kernel_size).reshape(B, -1, H_o2, W_o2)

            v1 = F.pad(v1, (pad1, pad1, pad1, pad1), mode='replicate')
            v2 = F.pad(v2, (pad2, pad2, pad2, pad2), mode='replicate')

            v1 = rearrange(v1, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.smk_size**2, h=H, w=W)
            v2 = rearrange(v2, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.kernel_size**2, h=H, w=W)

            x1 = einsum(attn1, v1, 'b g h w k, b g c h w k -> b g h w c')
            x2 = einsum(attn2, v2, 'b g h w k, b g c h w k -> b g h w c')

        x = torch.cat([x1, x2], dim=1)  # Join the first dimension
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)  # [B, 4, H, W, C] -> [B, 4×C, H, W]

        # Adjust the size of the input tensor x to the specified input_resolution
        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_resolution)  # [B, 4×C, H, W] -> [B, 4×C, H, W]

        x = self.fusion_proj(x)  # [B, 4×C, H, W] -> [B, 4×C, H, W]

        x = x + lepe  # [B, 4×C, H, W] -> [B, 4×C, H, W]
        x = self.se_layer(x)  # [B, 4×C, H, W] -> [B, 4×C, H, W]

        x = gate * x  # [B, 4×C, H, W] -> [B, 4×C, H, W]
        x = self.proj(x)  # [B, 4×C, H, W] -> [B, 4×C, H, W]

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))  # [B, 4×C, H, W] -> [B, 4×C, H, W]

        x = self.dwconv2(x)  # [B, 4×C, H, W] -> [B, 4×C, H, W]

        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))  # [B, 4×C, H, W] -> [B, 4×C, H, W]

        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, use_reentrant=False)
        else:
            x = self._forward_inner(x)
        return x

class DGCNet_B(nn.Module):
    def __init__(self,
                 in_chans=3,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 res_scale=False,
                 use_head1=False,
                 embed_dim=[32, 64, 128, 256],
                 ls_init_value=None,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 projection=256,
                 num_classes=121,
                 use_checkpoint=False,
                 ):
        super(DGCNet_B, self).__init__()
        self.use_head1 = use_head1
        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = stem(embed_dim[0], embed_dim[1])
        self.patch_embed3 = stem(embed_dim[1], embed_dim[-1])
        # self.patch_embed4 = stem(embed_dim[2], embed_dim[-1])

        self.conv1 = nn.Conv2d(embed_dim[-1], embed_dim[-1], kernel_size=1)

        self.DGC = ContMixBlock(
            dim=embed_dim[-1],
            kernel_size=kernel_size,
            smk_size=smk_size,
            num_heads=num_heads,
            ls_init_value=ls_init_value,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_gemm=use_gemm,
            deploy=deploy,
            use_checkpoint=use_checkpoint,
            mlp_ratio=mlp_ratio,
            res_scale=res_scale,
        )

        # classification head
        self.head = nn.Sequential(
            nn.BatchNorm2d(embed_dim[-1]),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim[-1], num_classes, kernel_size=1) if num_classes > 0 else nn.Identity(),
            nn.Sigmoid()
        )

        self.head1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 4096),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, projection),
            nn.SiLU(),
            nn.Dropout(0.3),
            # nn.Linear(2048, projection),
            # nn.SiLU(),
            # nn.Dropout(0.1),
            nn.Linear(projection, num_classes),
            nn.Sigmoid()
        )

    def reparm(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()

    def forward(self, x):
        x = self.patch_embed1(x)
        x = self.patch_embed2(x)
        x = self.patch_embed3(x)
        # x = self.patch_embed4(x)

        Identity = x
        Identity = self.conv1(Identity)

        x = self.DGC(x) + Identity

        if self.use_head1:
            x = self.head1(x)
        else:
            x = self.head(x).flatten(1)

        return x

if __name__ == "__main__":
    from thop import profile

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DGCNet_B(in_chans=3,
                kernel_size=7,
                 smk_size=5,
                 num_heads=4,
                 mlp_ratio=4,
                 res_scale=False,
                 use_head1=False,
                 embed_dim=[16, 32, 64, 128],
                 ls_init_value=None,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 projection=256,
                 num_classes=121,
                 use_checkpoint=False,
                     ).to(device)
    model.reparm()

    input = torch.randn(1, 3, 16, 16).to(device) # 示例输入
    flops, params = profile(model, inputs=(input, ))
    print(f"FLOPs: {flops / 1e6:.2f} M")
    print(f"Params: {params / 1e6:.2f} M")