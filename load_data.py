import h5py
import numpy as np
import pandas as pd
import torch
from scipy import interpolate


def winsorize(series, q1 = 0.1, q2 = 0.9):
    series_temp = series.copy()
    series = series.sort_values()
    q = series.quantile([q1, q2])
    
    return np.clip(series_temp, q.iloc[0], q.iloc[1])

    
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = xs[-1:].repeat((num_padding,1,1,1))
            y_padding = ys[-1:].repeat((num_padding,1,1))
            xs = torch.concat([xs, x_padding], axis=0)
            ys = torch.concat([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    
    
def downscale(n, chla_original, chla_imputed, sst, test_period_point):
    chla_original = np.nanmean(chla_original.reshape([chla_original.shape[0], chla_original.shape[1]//n, n, chla_original.shape[2]//n, n]), axis=(2,4))
    chla_imputed = np.nanmean(chla_imputed.reshape([chla_imputed.shape[0], chla_imputed.shape[1]//n, n, chla_imputed.shape[2]//n, n]), axis=(2,4))
    sst = np.nanmean(sst.reshape([sst.shape[0], sst.shape[1]//n, n, sst.shape[2]//n, n]), axis=(2,4))

    loc = np.where(np.isnan(chla_imputed[:test_period_point]).all(axis=0)==False)
    loc = np.stack((loc[0], loc[1]), axis=1)
    for loc_temp in loc:
        i = loc_temp[0]
        j = loc_temp[1]
        place1 = np.where(~np.isnan(sst[:, i, j]))[0]
        place2 = np.where(np.isnan(sst[:, i, j]))[0]
        if len(place2) > 0:
            tck = interpolate.splrep(place1, sst[place1, i, j])
            sst[place2, i, j] = interpolate.splev(place2, tck, der=0)
            
        chla_imputed[:, i, j] = winsorize(pd.Series(chla_imputed[:, i, j]))

    sst = -(sst-np.nanmean(sst[:test_period_point], axis=0))/np.nanstd(sst[:test_period_point], axis=0)*np.nanstd(chla_imputed[:test_period_point], axis=0)+np.nanmean(chla_imputed[:test_period_point], axis=0)
        
    variables = np.stack((chla_imputed, sst), axis=0)
    
    return chla_original, loc, variables


def load_data(data_path, test_period_point):
    chla_original = h5py.File(data_path+'modis_chla_month_4km_scs_1000m.mat', 'r')
    x = list(chla_original.keys())
    chla_original = np.array(chla_original[x[0]])
    
    chla_imputed = h5py.File(data_path+'modis_chla_month_4km_scs_1000m_imputed.mat', 'r')
    x = list(chla_imputed.keys())
    chla_imputed = np.array(chla_imputed[x[0]])
    chla_imputed = np.flip(chla_imputed, axis=1)

    sst = h5py.File(data_path+'modis_sst_month_4km_scs.mat', 'r')
    x = list(sst.keys())
    sst = np.array(sst[x[0]])[:216]

    n = 3
    chla_original, loc, variables = downscale(n, chla_original, chla_imputed, sst, test_period_point) 

    return chla_original, variables


def generate_data(variables, L1, L2, test_period_point, val_period_len):
    num = len(variables)

    X = torch.Tensor(variables).unfold(1, 2*L1, 1)
    X = torch.concatenate((X[0], X[1]), dim=3)
    X = X.reshape([X.shape[0], X.shape[1], X.shape[2], 2*num, L1])
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2], X.shape[3], X.shape[4])
    place = ~torch.isnan(X.sum(axis=(0,2,3)))
    X = X[:,place,:,:]
    X = X.transpose(1, 2)
    X = torch.cat((X, torch.zeros([X.shape[0], X.shape[1], X.shape[2], 1])), dim=3)
    # add month indicator
    X[:, 0, :, L1] = torch.Tensor((np.array(list(range(X.shape[0])))-1) % 12).repeat(X.shape[2], 1).T
    # X: (num_sample, input_dim, num_loc, input_length)
    X_train = X[1:test_period_point-2*L1-L2+1-val_period_len]
    X_val = X[test_period_point-2*L1-L2+1-val_period_len:test_period_point-2*L1-L2+1]
    X_test = X[test_period_point-2*L1-L2+1:-L2]
    
    Y = torch.Tensor(variables[0])
    Y = Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2])
    Y = Y[:,place]
    Y = Y.unsqueeze(-1)
    # Y: (num_sample, num_loc, output_length)
    Y_train = Y[2*L1+L2:test_period_point-val_period_len]
    Y_val = Y[test_period_point-val_period_len:test_period_point]
    Y_test = Y[test_period_point:]   

    X1 = torch.zeros(X_train.shape)
    X2 = torch.zeros(X_val.shape)
    X3 = torch.zeros(X_test.shape)
    Y1 = torch.zeros(Y_train.shape)
    Y2 = torch.zeros(Y_val.shape)
    Y3 = torch.zeros(Y_test.shape)
    # gather samples with the same prediction month
    for i in range(12):
        for j in range(len(X_train)//12):
            X1[i*(len(X_train)//12)+j] = X_train[j*12+i]
            Y1[i*(len(X_train)//12)+j] = Y_train[j*12+i]
        for j in range(len(X_val)//12):
            X2[i*(len(X_val)//12)+j] = X_val[j*12+i]
            Y2[i*(len(X_val)//12)+j] = Y_val[j*12+i]
        for j in range(len(X_test)//12):
            X3[i*(len(X_test)//12)+j] = X_test[j*12+i]
            Y3[i*(len(X_test)//12)+j] = Y_test[j*12+i]
    length = (len(X_train)//12)*12
    X_train = X1[:length]
    X_val = X2[:length]
    X_test = X3[:length]
    Y_train = Y1[:length]
    Y_val = Y2[:length]
    Y_test = Y3[:length]
    
    loc = place.reshape([variables.shape[2], variables.shape[3]])
    loc = np.where(loc==True)
    loc = np.stack((loc[0], loc[1]), axis=1)
    dist = torch.cdist(torch.Tensor(loc), torch.Tensor(loc), p=1, compute_mode='use_mm_for_euclid_dist_if_necessary')
    dist[dist <= 1] = 1
    dist[dist > 1] = 0
    dist = dist/dist.sum(axis=1)[:, np.newaxis]
    adj = dist

    print('number of locations:', len(loc))
    
    # each batch contains samples with the same prediction month
    batch_size = len(X_train)//12
    valid_batch_size = len(X_val)//12
    test_batch_size = len(X_test)//12
    
    dataloader = {}
    dataloader['X_train'] = X_train
    dataloader['Y_train'] = Y_train
    dataloader['X_val'] = X_val
    dataloader['Y_val'] = Y_val
    dataloader['X_test'] = X_test
    dataloader['Y_test'] = Y_test
    scaler = [dataloader['X_train'][:, 0, :, :12].mean(), dataloader['X_train'][:, 0, :, :12].std()]
    for i in range(num*2):
        mean = dataloader['X_train'][:, i, :, :12].mean()
        std = dataloader['X_train'][:, i, :, :12].std()
        for category in ['train', 'val', 'test']:
            dataloader['X_' + category][:, i, :, :12] = (dataloader['X_' + category][:, i, :, :12]-mean)/std
            
    dataloader['train_loader'] = DataLoader(dataloader['X_train'], dataloader['Y_train'], batch_size)
    dataloader['val_loader'] = DataLoader(dataloader['X_val'], dataloader['Y_val'], valid_batch_size)
    dataloader['test_loader'] = DataLoader(dataloader['X_test'], dataloader['Y_test'], test_batch_size)
    dataloader['scaler'] = scaler
    
    return loc, adj, dataloader, scaler