import torch.optim as optim
import math
import numpy as np
import torch
import metrics


class Trainer():
    def __init__(self, model, lr, wd, scaler, device, reg):
        self.scaler = scaler
        self.model = model
        self.optimizer = optim.Adam([{'params':([p for name, p in self.model.named_parameters() if 'characteristics' not in name]),'weight_decay':wd},{'params':([p for name, p in self.model.named_parameters() if 'characteristics' in name]),'weight_decay':0}], lr=lr)
        self.loss = metrics.masked_mae
        self.device = device
        self.reg = reg

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(input)
        predict = output*self.scaler[1]+self.scaler[0]
        real = torch.unsqueeze(real_val, dim=1)
        loss = self.loss(predict, real)
        for i in range(12):
#             loss = loss+self.reg*torch.trace(torch.matmul(torch.matmul(self.model.characteristics[:,i,:].T, torch.eye(len(self.model.characteristics)).to(self.device)-self.model.adj), self.model.characteristics[:,i,:]))
            loss = loss+self.reg*(self.model.adj*(self.model.characteristics[:,i,:].unsqueeze(dim=1)-self.model.characteristics[:,i,:].unsqueeze(dim=0)).square().sum(axis=2)).sum()
        for i in range(11):
            loss = loss+self.reg*(self.model.characteristics[:,i,:]-self.model.characteristics[:,i+1,:]).square().mean()
        loss = loss+self.reg*(self.model.characteristics[:,11,:]-self.model.characteristics[:,0,:]).square().mean()
        
        loss.backward()
        self.optimizer.step()
        mape = metrics.masked_mape(predict, real).item()
        rmse = metrics.masked_rmse(predict, real).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()

        output = self.model(input)
        predict = output*self.scaler[1]+self.scaler[0]
        real = torch.unsqueeze(real_val, dim=1)
        loss = self.loss(predict, real)
        mape = metrics.masked_mape(predict, real).item()
        rmse = metrics.masked_rmse(predict, real).item()
        return loss.item(), mape, rmse