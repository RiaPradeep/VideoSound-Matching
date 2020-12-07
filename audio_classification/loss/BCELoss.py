import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss_class = nn.BCELoss()
        self.d = nn.PairwiseDistance(p=2)
    
    def forward(self, pred, dummy, label):
        loss_1 = self.loss_class(pred, label.view(-1, 1).type(torch.float32)) 
        return loss_1.mean(), pred.reshape(-1)