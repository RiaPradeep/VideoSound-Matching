import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean(), distances

class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss = ContrastiveLoss(margin=128)
    
    def forward(self, enc1, enc2, label):  
        enc1 = torch.sigmoid(enc1) 
        enc2 = torch.sigmoid(enc2)
        loss, pred = self.loss(enc1, enc2, label)
        return loss, 1-pred.reshape(-1)
