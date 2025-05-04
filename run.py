import numpy as np
import torch
import time
import argparse
from load_data import *
from metrics import *
from trainer import Trainer
from STD_Hunter import *


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout rate')

parser.add_argument('--input_len',type=int,default=12,help='input sequence length')
parser.add_argument('--out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--pred_len',type=int,default=12,help='prediction horizon')
parser.add_argument('--in_dim',type=int,default=4,help='inputs dimension')
parser.add_argument('--rank',type=int,default=4,help='rank')
parser.add_argument('--channels',type=int,default=16,help='channels')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=1e-6,help='weight decay rate')
parser.add_argument('--reg',type=float,default=1e-5,help='regularization coefficient')

parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--save',type=str,default='save1',help='save path')
parser.add_argument('--runs',type=int,default=10,help='number of runs')

args = parser.parse_known_args()[0]


def train(args, test_period_point, chla_original, variables, loc, adj, dataloader, scaler):
    device = torch.device(args.device)
    model = STD_Hunter(args.num_nodes, device, adj=adj, dropout=args.dropout,
                  channels=args.channels, seq_length=args.input_len, 
                  in_dim=args.in_dim, out_dim=args.out_len, layers=args.layers, rank=args.rank)
    model = model.to(device)
    
    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, scaler, device, args.reg)

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
            trainy = torch.Tensor(y).to(device)
            metrics = engine.train(trainx, trainy)
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
            valy = torch.Tensor(y).to(device)
            metrics = engine.eval(valx, valy)
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
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            preds = engine.model(testx)
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    pred = yhat*scaler[1]+scaler[0]
    real = realy[:, :, 0]
    metrics = metric(pred, real)
    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(1, metrics[0], metrics[1], metrics[2]))
    mae.append(metrics[0])
    mape.append(metrics[1])
    rmse.append(metrics[2])

    pred_temp = torch.zeros(pred.shape)
    for i in range(12):
        for j in range(len(pred_temp)//12):
            pred_temp[j*12+i] = pred[i*(len(pred_temp)//12)+j]
    prediction = np.full(chla_original.shape, np.nan)
    prediction[test_period_point:,loc[:,0],loc[:,1]]=pred_temp

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
    loc, adj, dataloader, scaler = generate_data(variables, L1, L2, test_period_point, val_period_len)
        
    parser.add_argument('--num_nodes',type=int,default=len(loc),help='number of nodes/variables')
    args = parser.parse_known_args()[0]
        
    device = torch.device(args.device)
    for category in ['train', 'val', 'test']:
        dataloader['X_' + category]=dataloader['X_' + category].to(device)
        dataloader['Y_' + category]=dataloader['Y_' + category].to(device)
    scaler = torch.Tensor(scaler).to(device)
    adj = adj.to(device)
    
    mae_list = []
    rmse_list = []
    mape_list = []
    corr_list = []
    for i in range(args.runs):
        print('start run', i+1)
        mae, rmse, mape, corr = train(args, test_period_point, chla_original, variables, loc, adj, dataloader, scaler)
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
