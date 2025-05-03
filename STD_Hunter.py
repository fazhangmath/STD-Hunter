import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import numbers
import torch.nn.functional as F


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

    def forward(self, input):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

    
class STD_Hunter(nn.Module):
    def __init__(self, num_nodes, device, adj, dropout=0.3, channels=32, seq_length=12, in_dim=2, out_dim=1, layers=3, rank=4):
        super(STD_Hunter, self).__init__()
        self.dropout = dropout
        self.channels = channels
        self.adj = adj
        self.device = device
        self.seq_length = seq_length
        self.norm = nn.ModuleList()
        self.skip_convs = nn.ModuleList()    
        self.rank = rank
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=channels, kernel_size=(1, 1))
        self.layers = layers

        self.characteristics = nn.Parameter(torch.empty(size=(num_nodes, 12, self.rank)))
        nn.init.xavier_uniform_(self.characteristics.data, gain=1.414)
        
        self.conv_w = nn.Linear(self.rank, int(channels/4)*channels*18*layers)
        self.conv_b = nn.Linear(self.rank, int(channels/4)*4*layers)
        self.gate_w = nn.Linear(self.rank, int(channels/4)*channels*18*layers)
        self.gate_b = nn.Linear(self.rank, int(channels/4)*4*layers)
        
        self.res_w = nn.Linear(self.rank, channels*channels*layers)
        self.res_b = nn.Linear(self.rank, channels*layers)     
                            
        for j in range(layers):
            self.norm.append(LayerNorm((channels, num_nodes, seq_length),elementwise_affine=True))
            self.skip_convs.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, seq_length), groups=channels))

        self.end_conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
        
        self.idx = torch.arange(num_nodes).to(device)
        self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=channels, kernel_size=(1, seq_length), bias=True)
        self.skipE = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, seq_length), groups=channels)

    def forward(self, input):
        # the prediction month of this batch
        month = input[0, 0, 0, -1]
        input = input[:, :, :, 0:self.seq_length]
        
        characteristics = self.characteristics
        # graph convolutional layer on spatiotemporal characteristics
        adj_month = np.diag(np.ones(12))/3
        np.fill_diagonal(adj_month[1:], 1/3)
        np.fill_diagonal(adj_month[:, 1:], 1/3)
        adj_month[0, 11] = 1/3
        adj_month[11, 0] = 1/3
        adj_month = torch.Tensor(adj_month).to(self.device)
        characteristics = torch.matmul(adj_month, characteristics)
        characteristics = torch.matmul(self.adj, characteristics[:, torch.tensor(month, dtype=torch.long)])

        # generate parameters for TC layers using characteristics
        conv_w_all = self.conv_w(characteristics)
        conv_b_all = self.conv_b(characteristics)
        gate_w_all = self.gate_w(characteristics)
        gate_b_all = self.gate_b(characteristics)
        res_w_all = self.res_w(characteristics)
        res_b_all = self.res_b(characteristics)
        
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            channel_num = int(self.channels/4)
            param_num = channel_num*self.channels 
            conv_w = conv_w_all[:, (i*18)*param_num:(i+1)*18*param_num].view(x.shape[2],channel_num,x.shape[1],1,18).flatten(0, 1)
            conv_b = conv_b_all[:, i*4*channel_num:(i+1)*4*channel_num].view(x.shape[2], channel_num*4).flatten(0, 1)
            gate_w = gate_w_all[:, (i*18)*param_num:(i+1)*18*param_num].view(x.shape[2],channel_num,x.shape[1],1,18).flatten(0, 1)
            gate_b = gate_b_all[:, i*4*channel_num:(i+1)*4*channel_num].view(x.shape[2], channel_num*4).flatten(0, 1) 
            filter = torch.zeros(x.shape).to(self.device)
            gate = torch.zeros(x.shape).to(self.device) 
            shape = x.shape
            group_num = x.shape[2]
            x = x.transpose(1,2).reshape(x.shape[0], x.shape[1]*x.shape[2], 1, x.shape[3])
            
            filter[:,0:channel_num] = F.conv2d(F.pad(x, (0,1,0,0), mode='circular'),conv_w[...,:2],conv_b[:int(len(conv_b)/4)],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)
            filter[:,channel_num:2*channel_num] = F.conv2d(F.pad(x, (0,2,0,0), mode='circular'),conv_w[...,2:5],conv_b[int(len(conv_b)/4):int(len(conv_b)/4)*2],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)  
            filter[:,2*channel_num:3*channel_num] = F.conv2d(F.pad(x, (0,5,0,0), mode='circular'),conv_w[...,5:11],conv_b[int(len(conv_b)/4)*2:int(len(conv_b)/4)*3],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)  
            filter[:,3*channel_num:4*channel_num] = F.conv2d(F.pad(x, (0,6,0,0), mode='circular'),conv_w[...,11:],conv_b[int(len(conv_b)/4)*3:],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)  
            filter = torch.tanh(filter)        
            
            gate[:,0:channel_num] = F.conv2d(F.pad(x, (0,1,0,0), mode='circular'),gate_w[...,:2],gate_b[:int(len(conv_b)/4)],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)
            gate[:,channel_num:2*channel_num] = F.conv2d(F.pad(x, (0,2,0,0), mode='circular'),gate_w[...,2:5],gate_b[int(len(conv_b)/4):int(len(conv_b)/4)*2],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)  
            gate[:,2*channel_num:3*channel_num] = F.conv2d(F.pad(x, (0,5,0,0), mode='circular'),gate_w[...,5:11],gate_b[int(len(conv_b)/4)*2:int(len(conv_b)/4)*3],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)  
            gate[:,3*channel_num:4*channel_num] = F.conv2d(F.pad(x, (0,6,0,0), mode='circular'),gate_w[...,11:],gate_b[int(len(conv_b)/4)*3:],groups=group_num).reshape((shape[0], shape[2], channel_num, shape[3])).transpose(1,2)  
            gate = torch.sigmoid(gate)                       

            x = filter*gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.skip_convs[i](x)
            # skip connection
            skip = s+skip              
            param_num = int(self.res_w.weight.shape[0]/self.layers)
            res_w = res_w_all[:, i*param_num:(i+1)*param_num].view(x.shape[2],x.shape[1],x.shape[1],1,1).flatten(0, 1)
            res_b = res_b_all[:, i*x.shape[1]:(i+1)*x.shape[1]].view(x.shape[2], x.shape[1]).flatten(0, 1)
            x = F.conv2d(x.transpose(1,2).reshape(x.shape[0], x.shape[1]*x.shape[2], 1, x.shape[3]),res_w,res_b,groups=x.shape[2]).reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3]).transpose(1,2)
            # residual connection
            x = x + residual
            
            x = self.norm[i](x)

        skip = F.relu(self.skipE(x)+skip)
        x = F.relu(self.end_conv_1(skip))
        x = self.end_conv_2(x)
        return x