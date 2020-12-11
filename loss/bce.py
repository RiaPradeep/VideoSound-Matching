import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_class = nn.BCELoss()
    
    def forward(self, pred, dummy, label):
        loss = self.loss_class(pred, label.view(-1, 1).type(torch.float32)) 
        return loss.mean(), pred.reshape(-1)
