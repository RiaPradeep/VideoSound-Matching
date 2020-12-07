import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss_class =  nn.BCELoss()
        self.d = nn.PairwiseDistance(p=2)
        self.cos = nn.CosineSimilarity()
    
    def forward(self, pred, audio, label, l=0.1):
        loss_1 = self.loss_class(pred, label.view(-1, 1).type(torch.float32)) 
        cosine = self.cos(audio[0], audio[1])
        loss_21 = label * (1- cosine)
        loss_22 = (1-label) * torch.relu(cosine - 0.2)
        loss = loss_21 + loss_22 + l * loss_1
        #print(torch.min(pred), torch.max(pred)) 
        return loss.mean(), pred.reshape(-1)
