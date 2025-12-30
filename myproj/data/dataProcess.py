import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label[:, 0])

    def __getitem__(self, idx):
        data = self.data[idx, :, :, :]
        label = self.label[idx, :]
        return data, label


def Train_DataLoader(filename, test_size, data='ECM', use_TLdata=False):
    Data = h5py.File(filename, 'r')
    angles = np.transpose(np.array(Data['angles']))
    Rxx = np.array(Data[data])

    [SNRs, n, chan, M, N] = Rxx.shape # n: SNR levels
    X_data = Rxx.reshape([SNRs * n, chan, N, M]) # reshape to [num of total samples, channels, height, width]

    mlb = MultiLabelBinarizer()
    y_Train_encoded = mlb.fit_transform(angles)  # Constructing a multi label classification matrix

    if use_TLdata:
        Y_Labels = y_Train_encoded
    else:
        Y_Labels = np.tile(y_Train_encoded, reps=(SNRs,1))  # Since there are SNR scenes in each angle combination, ytrain_encoded will copy SNR along the first dimension to form the label of the overall training set

    xTrain, xVal, yTrain, yVal = train_test_split(X_data, Y_Labels, test_size=test_size, random_state=42)
    xTrain = torch.tensor(xTrain, dtype=torch.float)
    xVal = torch.tensor(xVal, dtype=torch.float)
    yTrain = torch.tensor(yTrain, dtype=torch.float)
    yVal = torch.tensor(yVal, dtype=torch.float)

    return xTrain, xVal, yTrain, yVal