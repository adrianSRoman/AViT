import torch

def ce_loss():
    return torch.nn.CrossEntropyLoss()

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()
