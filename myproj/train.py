import sys
import os
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from model.Dynamic_Global_ConvolutionNet_B import *
from model.regularizer import *
from model.ASL import AsymmetricLoss
from data.dataProcess import *
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# data file (Change to your own path)
filename_data_Pre_train = '/home/hipeson/zwj/myProjects/data/ECM_k=2.h5'
filename_data_Fine_tuning = '/home/hipeson/zwj/myProjects/data/SCM_k=2.h5'
# file to save models (Change to your own path)
filename_save_root = '/home/hipeson/zwj/myProjects/models/'
filename_save_model_Pre_train = filename_save_root + 'Pre_train_cnn/'
filename_save_model_Fine_tuning =filename_save_root + 'Fine_tuning_cnn_k=2/'
# file to save logs (Change to your own path)
filename_logs_root = '/home/hipeson/zwj/myProjects/Result/'
writer_Pre_train = SummaryWriter(filename_logs_root+"Pre_train_cnn")
writer_Fine_tuning = SummaryWriter(filename_logs_root+"Fine_tuning_cnn")

# Pre_train parameters
epochs_Pre_train = 150
batch_size_Pre_train = 64
Cov_Matrix_type_Pre_train = 'ECM'
# Fine_tuning parameters
epochs_Fine_tuning = 10
batch_size_Fine_tuning = 64
Cov_Matrix_type_Fine_tuning = 'SCM'
# dataset parameters
Validation_set_size = 0.2  # 80% for Training set and 20% for Validation set
# DGCNet model
model = DGCNet_B(in_chans=3,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=4,
                 mlp_ratio=4,
                 res_scale=True,
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
               ).to(device)
model.reparm()


def one_epoch(model, optimizer, lr, criterion, L1_Regularizer, dataloader, device, epoch_now, epoch_all, is_train,
              is_L1_Regularizer):
    if is_train == True:
        model.train()
    else:
        model.eval()

    mean_loss = torch.zeros(1).to(device)
    dataloader = tqdm(dataloader, desc=f'Epoch {epoch_now + 1}/{epoch_all}', file=sys.stdout)

    for step, data in enumerate(dataloader):
        x_batch, y_batch = data
        ouputs_train = model(x_batch.to(device))
        loss = criterion(ouputs_train, y_batch.to(device).to(torch.float32))

        if is_L1_Regularizer == True:
            for model_param_name, model_param_value in model.named_parameters():
                if model_param_name in ['head.weight']:
                    loss = L1_Regularizer.regularized_param(param_weights=model_param_value,
                                                            reg_loss_function=loss)

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        if is_train == True:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            dataloader.desc = "[Training epoch {}] mean loss:{}".format(epoch_now,
                                                                        round(mean_loss.item(), 6)) + ' lr:{}'.format(lr)
        else:
            dataloader.desc = "[Val epoch {}] mean loss:{}".format(epoch_now,
                                                                   round(mean_loss.item(), 6)) + ' lr:{}'.format(lr)
    return mean_loss.item()


def Pre_Train():
    print('=' * 80)
    print('Pre-training started.')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

    # Parameters of early stop method
    patience = 50  # Number of epochs with no performance improvement on the validation set
    min_delta = 0.0005  # Minimum change considered to be improved
    best_val_loss = float('inf')
    counter = 0  # Record the number of epoch without improvement
    early_stop = False

    xTrain, xVal, yTrain, yVal = Train_DataLoader(filename_data_Pre_train, Validation_set_size,
                                                         Cov_Matrix_type_Pre_train, use_TLdata=False) # obtain train data

    criterion = AsymmetricLoss().to(device) # loss function
    L1_Regularizer = L1Regularizer(model, lambda_reg=5e-5) # L1 norm
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False) # optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=5)

    Train_dataloader = DataLoader(dataset=MyDataset(xTrain, yTrain), batch_size=batch_size_Pre_train, shuffle=True)
    Val_dataloader = DataLoader(dataset=MyDataset(xVal, yVal), batch_size=batch_size_Pre_train, shuffle=True)
    Val_loss_min = 99999.0

    for epoch in range(epochs_Pre_train):
        # If an early stop is triggered, end the training in advance

        Train_loss = one_epoch(model, optimizer, optimizer.param_groups[0]['lr'], criterion, L1_Regularizer,
                               Train_dataloader, device, epoch, epochs_Pre_train, is_train=True, is_L1_Regularizer=True)

        Val_loss = one_epoch(model, optimizer, optimizer.param_groups[0]['lr'], criterion, L1_Regularizer,
                             Val_dataloader, device, epoch, epochs_Pre_train, is_train=False, is_L1_Regularizer=True)

        scheduler.step(Val_loss)

        # Record training
        writer_Pre_train.add_scalar("Train_loss", Train_loss, epoch)
        writer_Pre_train.add_scalar("Val_loss", Val_loss, epoch)
        writer_Pre_train.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # Early stop logic
        if Val_loss < best_val_loss - min_delta:
            best_val_loss = Val_loss
            counter = 0  # Reset counter
            # Save best model
            torch.save(model, filename_save_model_Pre_train + "/model_best.pth")
            print(f"Validation loss improved to {Val_loss:.6f}, saving best model")
        else:
            counter += 1
            print(f"Validation loss did not improve for {counter} epoch(s)")
            if counter >= patience:
                early_stop = True
                print(f"Early stopping after {patience} epochs without improvement")

        # Save the model every 10 rounds
        if (epoch + 1) % 10 == 0:
            torch.save(model, filename_save_model_Pre_train + "/model_{}.pth".format(epoch + 1))

        # Keep the original best model saving logic
        if Val_loss_min > Val_loss:
            Val_loss_min = Val_loss

    print('Pre-training finished')
    print('=' * 80)


def Fine_tuning():
    print('='*80)
    print("Fine-tuning started.")

    xTrain, xVal, yTrain, yVal = Train_DataLoader(filename_data_Fine_tuning, Validation_set_size,
                                                     Cov_Matrix_type_Fine_tuning, use_TLdata=True)

    # initialize Fine_tuning model with  parameters of Pre_train model
    checkpoint = torch.load(filename_save_model_Pre_train + "model_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.state_dict())

    criterion = AsymmetricLoss().to(device)
    L1_Regularizer = L1Regularizer(model, lambda_reg=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=5)

    Train_dataloader = DataLoader(dataset=MyDataset(xTrain, yTrain), batch_size=batch_size_Fine_tuning, shuffle=True)
    Val_dataloader = DataLoader(dataset=MyDataset(xVal, yVal), batch_size=batch_size_Fine_tuning, shuffle=True)

    Val_loss_min = 99999.0

    for epoch in range(epochs_Fine_tuning):

        Train_loss = one_epoch(model, optimizer, optimizer.param_groups[0]['lr'], criterion, L1_Regularizer,
                               Train_dataloader, device, epoch,epochs_Fine_tuning, is_train=True, is_L1_Regularizer=True)

        Val_loss = one_epoch(model, optimizer, optimizer.param_groups[0]['lr'], criterion, L1_Regularizer,
                             Val_dataloader, device,  epoch,epochs_Fine_tuning, is_train=False, is_L1_Regularizer=True)

        scheduler.step(Val_loss)

        writer_Fine_tuning.add_scalar("Train_loss", Train_loss, epoch)
        writer_Fine_tuning.add_scalar("Val_loss", Val_loss, epoch)
        writer_Fine_tuning.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(model, filename_save_model_Fine_tuning + "/model_{}.pth".format(epoch + 1))
        if Val_loss_min > Val_loss:
            Val_loss_min = Val_loss
            torch.save(model, filename_save_model_Fine_tuning + "/model_best.pth")

    print("Fine-tuning is done.")
    print('='*80)


if __name__ == "__main__":
    if os.path.exists(filename_save_model_Pre_train) is False:
        os.makedirs(filename_save_model_Pre_train)
    if os.path.exists(filename_save_model_Fine_tuning) is False:
        os.makedirs(filename_save_model_Fine_tuning)
    Pre_Train()
    Fine_tuning()
