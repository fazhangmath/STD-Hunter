import numpy as np
import torch


def masked_mae(preds, labels):
    mask = ~torch.isnan(labels)
    mask = mask.float()
    loss = torch.abs(preds - labels)
    loss = loss.nansum()/mask.sum()
    return loss


def masked_rmse(preds, labels):
    mask = ~torch.isnan(labels)
    mask = mask.float()
    loss = (preds - labels) ** 2
    loss = loss.nansum()/mask.sum()
    return torch.sqrt(loss)


def masked_mape(preds, labels):
    mask = ~torch.isnan(labels)
    mask = mask.float()
    loss = torch.abs(preds - labels) / labels
    loss = loss.nansum()/mask.sum()
    return loss


def metric(pred, real):
    mae = masked_mae(pred, real).item()
    mape = masked_mape(pred, real).item()
    rmse = masked_rmse(pred, real).item()
    return mae, mape, rmse