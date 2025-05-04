import numpy as np
import torch
import torch.nn as nn
import os
import time
from load_data import *


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out


def train(chla_original, variables, loc, device):    
    X = torch.Tensor(variables[:, :, loc[:,0], loc[:,1]]).transpose(0,2).unfold(1, 2*L1, 1)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
    X_train = X[:, 1:test_period_point-2*L1-L2+1-val_period_len]
    X_test = X[:, test_period_point-2*L1-L2+1:-L2]
    X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 1, X_train.shape[2]).to(device)
    X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1], 1, X_test.shape[2]).to(device)
    Y = torch.Tensor(variables[0][:, loc[:, 0], loc[:, 1]]).transpose(0,1)
    Y = Y.reshape(Y.shape[0],Y.shape[1],1)
    Y_train = Y[:, 2*L1+L2:test_period_point-val_period_len]
    Y_test = Y[:, test_period_point:]   
    Y_train = Y_train.reshape(Y_train.shape[0]*Y_train.shape[1] ,1).to(device)
    Y_test = Y_test.reshape(Y_test.shape[0]*Y_test.shape[1], 1).to(device)
    
    input_dim = X_train.shape[-1]
    hidden_dim = 16
    output_dim = 1
    model = LSTMModel(input_dim, hidden_dim, output_dim)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = int(X_train.size(0))
    epoch = 1000
    num_batches = int(X_train.size(0) / batch_size)
    for i in range(epoch):
        for batch in range(num_batches):
            optimizer.zero_grad()
            
            start_idx = batch * batch_size
            end_idx = (batch + 1) * batch_size
            inputs = X_train[start_idx:end_idx]
            targets = Y_train[start_idx:end_idx]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
    prediction = np.full(chla_original.shape, np.nan)
    pred = model(X_test).squeeze(-1).detach().cpu().numpy()
    pred = pred.reshape(len(loc), pred.shape[0]//len(loc)).T
    prediction[test_period_point:,loc[:,0],loc[:,1]] = pred

    mae = []
    rmse = []
    mape = []
    corr = []
    for k in range(len(loc)):
        i = loc[k][0]
        j = loc[k][1]
        real = chla_original[test_period_point:, i, j][~np.isnan(chla_original[test_period_point:, i, j])]
        pred = prediction[test_period_point:, i, j][~np.isnan(chla_original[test_period_point:, i, j])]
        error = real-pred
        mae.append(np.mean(np.abs(error)))
        rmse.append(np.sqrt((error ** 2).sum()/len(error)))
        mape.append(np.mean(np.abs(error)/real))
        corr.append(np.corrcoef(real, pred)[0][1])
        
    return np.mean(mae), np.mean(rmse), np.mean(mape), np.mean(corr)
    
    
if __name__ == "__main__":
    data_path='./'
    
    test_period_point = 168
    chla_original, variables = load_data(data_path, test_period_point)
    loc = np.where(np.isnan(variables[0][:test_period_point]).all(axis=0)==False)
    loc = np.stack((loc[0], loc[1]), axis=1)

    L1 = 12
    L2 = 12
    val_period_len = 24
    runs = 10
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    mae_list = []
    rmse_list = []
    mape_list = []
    corr_list = []
    for i in range(runs):
        print('start run', i+1)
        mae, rmse, mape, corr = train(chla_original, variables, loc, device)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        corr_list.append(corr)

    mae_list = np.array(mae_list)
    rmse_list = np.array(rmse_list)
    mape_list = np.array(mape_list)
    corr_list = np.array(corr_list)

    print('test|\MAE-mean\RMSE-mean\MAPE-mean\CORR-mean\MAE-std\RMSE-std\MAPE-std\CORR-std')
    log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(i+1, mae_list.mean(), rmse_list.mean(), mape_list.mean(), corr_list.mean(), mae_list.std(), rmse_list.std(), mape_list.std(), corr_list.std()))
