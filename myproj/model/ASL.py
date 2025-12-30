import torch.nn as nn
import torch


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        use_sigmoid=False,
    ):
        """AsymmetricLoss implements an asymmetric loss function, which is
        useful for imbalanced classification problems.

        It has two separate parameters to control the weight of positive and
        negative classes (gamma_pos and gamma_neg).
        It also includes an option to clip the prediction values for the
        negative class to ignore easy negative classes.

        Args:
            gamma_neg (float, optional): The focusing parameter for the negative
                class, default is 4.
            gamma_pos (float, optional): The focusing parameter for the positive
                class, default is 1.
            clip (float, optional): The clipping parameter for the negative
                class, default is 0.05.
            eps (float, optional): A small constant to prevent log of zero,
                default is 1e-8.
            disable_torch_grad_focal_loss (bool, optional): A flag to disable
                the gradient computation for the focal loss part, default is
                True.

        References:
            Ridnik, Tal, Emanuel Ben-Baruch, Nadav Zamir, Asaf Noy, Itamar
            Friedman, Matan Protter, and Lihi Zelnik-Manor. “Asymmetric Loss for
            Multi-Label Classification,” 82–91, 2021.
        """
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, x, y):
        """ The forward method calculates the asymmetric loss for the given
        inputs.

        Args:
            x (torch.Tensor): The input logits.
            y (torch.Tensor): The targets (multi-label binarized vector).

        Returns:
            torch.Tensor: The calculated loss.
        """

        # Calculating Probabilities
        if self.use_sigmoid:
            x_sigmoid = torch.sigmoid(x)
        else:
            x_sigmoid = x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (probability shifting)
        # Asymmetric tailoring: tailoring the probability of negative samples to avoid the training process dominated by too simple negative samples
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation   Loss = -[y * log(p) + (1 - y) * log(1 - p)]
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Gradient is disabled because gradient is not required for weight calculation
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean(1).sum()