import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss = torch.nn.TripletMarginLoss(margin=700)
        self.d = nn.PairwiseDistance(p=2)
    
    def forward(self, audio1_enc, audio2_enc, video_enc, label):
        zero_label_loss = self.loss(video_enc, audio1_enc, audio2_enc)
        one_label_loss = self.loss(video_enc, audio2_enc, audio1_enc)
        loss = (1-label) * zero_label_loss + label * one_label_loss
        dist1 = self.d(audio1_enc, video_enc).reshape(-1, 1)
        dist2 = self.d(audio2_enc, video_enc).reshape(-1, 1)

        distances = torch.cat([dist1, dist2], dim=1)
        #print(torch.max(dist1), torch.min(dist1), torch.max(dist2), torch.min(dist2))
        return loss.sum(), distances
