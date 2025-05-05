import numpy as np
import torch
import time
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import *


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
    

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_bool',type=str_to_bool, default=True,help='whether to add graph convolution layer')
parser.add_argument('--aptonly',type=str_to_bool, default=False,help='whether only adaptive adj')
parser.add_argument('--addaptadj',type=str_to_bool, default=False,help='whether add adaptive adj')
parser.add_argument('--randomadj',type=str_to_bool, default=False,help='whether random initialize adaptive adj')
parser.add_argument('--input_len',type=int,default=12,help='input sequence length')
parser.add_argument('--pred_len',type=int,default=12,help='prediction horizon')
parser.add_argument('--seq_length',type=int,default=1,help='')
parser.add_argument('--nhid',type=int,default=8,help='')
parser.add_argument('--batch_size',type=int,default=12,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.01,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--save',type=str,default='./save_Graph_WaveNet',help='save path')
parser.add_argument('--runs',type=int,default=10,help='number of runs')

args = parser.parse_known_args()[0]


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
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

    
class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

    
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

    
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse


def generate_data_Graph_WaveNet(args, variables, L1, L2, test_period_point, val_period_len):
    num = len(variables)

    X = torch.Tensor(variables).unfold(1, 2*L1, 1)
    X = torch.concatenate((X[0], X[1]), dim=3)
    X = X.reshape([X.shape[0], X.shape[1], X.shape[2], 2*num, L1])
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2], X.shape[3], X.shape[4])
    place = ~torch.isnan(X.sum(axis=(0,2,3)))
    X = X[:,place,:,:]
    X = X.transpose(1, 2)
    X = X.transpose(1, 3)
    X_train = X[1:test_period_point-2*L1-L2+1-val_period_len]
    X_val = X[test_period_point-2*L1-L2+1-val_period_len:test_period_point-2*L1-L2+1]
    X_test = X[test_period_point-2*L1-L2+1:-L2]
    
    Y = torch.Tensor(variables[0])
    Y = Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2])
    Y = Y[:,place]
    Y = Y.unsqueeze(-1).unsqueeze(1)
    Y_train = Y[2*L1+L2:test_period_point-val_period_len]
    Y_val = Y[test_period_point-val_period_len:test_period_point]
    Y_test = Y[test_period_point:]   
  
    loc = place.reshape([variables.shape[2], variables.shape[3]])
    loc = np.where(loc==True)
    loc = np.stack((loc[0], loc[1]), axis=1)
    dist = torch.cdist(torch.Tensor(loc), torch.Tensor(loc), p=1, compute_mode='use_mm_for_euclid_dist_if_necessary')
    dist[dist <= 1] = 1
    dist[dist > 1] = 0
    dist = dist/dist.sum(axis=1)[:, np.newaxis]
    adj = dist

    print('number of locations:', len(loc))
    
    batch_size = args.batch_size
    
    dataloader = {}
    dataloader['X_train'] = X_train
    dataloader['Y_train'] = Y_train
    dataloader['X_val'] = X_val
    dataloader['Y_val'] = Y_val
    dataloader['X_test'] = X_test
    dataloader['Y_test'] = Y_test
    scaler = StandardScaler(mean=dataloader['X_train'][:, 0:12, :, 0].mean(), std=dataloader['X_train'][:, 0:12, :, 0].std())
    for i in range(num*2):
        mean = dataloader['X_train'][:, i, :, :12].mean()
        std = dataloader['X_train'][:, i, :, :12].std()
        for category in ['train', 'val', 'test']:
            dataloader['X_' + category][:, i, :, :12] = (dataloader['X_' + category][:, i, :, :12]-mean)/std
            
    dataloader['train_loader'] = DataLoader(dataloader['X_train'], dataloader['Y_train'], batch_size)
    dataloader['val_loader'] = DataLoader(dataloader['X_val'], dataloader['Y_val'], batch_size)
    dataloader['test_loader'] = DataLoader(dataloader['X_test'], dataloader['Y_test'], batch_size)
    dataloader['scaler'] = scaler
    
    return loc, adj, dataloader, scaler


def train(args, test_period_point, chla_original, loc, adj, dataloader, scaler):
    device = torch.device(args.device)
    adj = [adj]
    supports = [torch.tensor(i).to(device) for i in adj]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)
    
    print(args)
    
    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1,args.epochs+1):
        #train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
        t2 = time.time()
        train_time0=t2-t1
        train_time.append(train_time0)
        
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            valx = valx.transpose(1, 3)
            valy = torch.Tensor(y).to(device)
            valy = valy.transpose(1, 3)
            metrics = engine.eval(valx, valy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        t2 = time.time()
        val_time0=t2-t1
        val_time.append(val_time0)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, train_time0),flush=True)        
        log = 'Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Inference Time: {:.4f}/epoch'
        print(log.format(i, mvalid_loss, mvalid_mape, mvalid_rmse, val_time0),flush=True)

        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), args.save +'.pth')
            minl = mvalid_loss

    print('Average Training Time: {:.4f} secs/epoch'.format(np.mean(train_time)))
    print('Average Inference Time: {:.4f} secs'.format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save +'.pth'))
    print('Training finished')
    print('The valid loss on best model is', str(round(his_loss[bestid],4)))

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['Y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    pred = scaler.inverse_transform(yhat)
    real = realy[:, :, 0]
    metrics = metric(pred, real)
    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(1, metrics[0], metrics[1], metrics[2]))

    prediction = np.full(chla_original.shape, np.nan)
    prediction[test_period_point:,loc[:,0],loc[:,1]] = pred.detach().cpu().numpy()

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
    
    # starting point of test period
    test_period_point = 168
    chla_original, variables = load_data(data_path, test_period_point)
    
    L1 = args.input_len
    L2 = args.pred_len
    # length of validation period
    val_period_len = 24
    loc, adj, dataloader, scaler = generate_data_Graph_WaveNet(args, variables, L1, L2, test_period_point, val_period_len)
    
    parser.add_argument('--in_dim',type=int,default=2*len(variables),help='inputs dimension')
    parser.add_argument('--num_nodes',type=int,default=len(loc),help='number of nodes/variables')
    args = parser.parse_known_args()[0]
        
    device = torch.device(args.device)
    for category in ['train', 'val', 'test']:
        dataloader['X_' + category]=dataloader['X_' + category].to(device)
        dataloader['Y_' + category]=dataloader['Y_' + category].to(device)
    adj = adj.to(device)
    
    mae_list = []
    rmse_list = []
    mape_list = []
    corr_list = []
    for i in range(args.runs):
        print('start run', i+1)
        mae, rmse, mape, corr = train(args, test_period_point, chla_original, loc, adj, dataloader, scaler)
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
