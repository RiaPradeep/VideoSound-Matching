import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_class = nn.BCELoss()
        self.d = nn.PairwiseDistance(p=2)
        self.cos = nn.CosineSimilarity()
    
    def forward(self, pred, audio, label, reg=0.1):
        loss_bce = self.loss_class(pred, label.view(-1, 1).type(torch.float32)) 
        cosine = self.cos(audio[0], audio[1])
        loss_pos = label * (1 - cosine)
        loss_neg = (1 - label) * torch.relu(cosine - 0.2)
        loss = loss_pos + loss_neg + reg * loss_bce
        return loss.mean(), pred.reshape(-1)
