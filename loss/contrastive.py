import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_class = ContrastiveLoss()
    
    def forward(self, pred, dummy, label):
        loss = self.loss_class(pred, label.view(-1, 1).type(torch.float32)) 
        return loss.mean(), pred.reshape(-1)
