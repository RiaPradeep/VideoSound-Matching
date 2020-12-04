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
        #self.loss = ContrastiveLoss(margin=700)
        self.loss_class =  nn.BCEWithLogitsLoss()
        self.d = nn.PairwiseDistance(p=2)
        self.linear = nn.Sequential(nn.Linear(800, 1))
    
    def forward(self, audio1_enc, video_enc, label):
        #zero_label_loss = self.loss(video_enc, audio1_enc, 1-label)
        #one_label_loss = self.loss(video_enc, audio2_enc, label)
        #loss_1 = zero_label_loss + one_label_loss
        #dist1 = self.d(audio1_enc, video_enc).reshape(-1, 1)
        #dist2 = self.d(audio2_enc, video_enc).reshape(-1, 1)
        #distances = torch.cat([dist1, dist2], dim=1)
        #print(audio1_enc, audio2_enc, video_enc)
        output = torch.cat((video_enc, audio1_enc), 1).to(video_enc.device)
        pred = self.linear(output)
        loss_1 = self.loss_class(pred, label.view(-1, 1).type(torch.float32))   
        #loss_2 = self.loss_class(audio2_enc, label)
        #loss = loss_1 + loss_2
        #distances = torch.cat([loss_1, loss_2], dim=1)
        
        #loss = loss_1/200000 + loss_2
        return loss_1.sum(), pred.reshape(-1)
        #, distances
