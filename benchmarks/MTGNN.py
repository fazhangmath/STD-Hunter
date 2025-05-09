import numpy as np
import torch
import time
import numbers
from torch.nn import init
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

parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=False,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=16,help='skip channels')
parser.add_argument('--end_channels',type=int,default=16,help='end channels')

parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--pred_len',type=int,default=12,help='prediction horizon')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')

parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='save_MTGNN',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

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


class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, input, real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, idx=idx)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
        if self.cl:
            loss = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        # mae = masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        self.iter += 1
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

    
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
    
    
class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    

def generate_data_Graph_MTGNN(args, variables, L1, L2, test_period_point, val_period_len):
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
    predefined_A = adj

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)
    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)

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
    
    L1 = args.seq_in_len
    L2 = args.pred_len
    # length of validation period
    val_period_len = 24
    loc, adj, dataloader, scaler = generate_data_Graph_MTGNN(args, variables, L1, L2, test_period_point, val_period_len)
    
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
