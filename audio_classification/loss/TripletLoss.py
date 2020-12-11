import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss = torch.nn.TripletMarginLoss(margin=128)
        self.d = nn.PairwiseDistance(p=2)
    
    def forward(self, audio1_enc, audio2_enc, video_enc):
        loss = self.loss(video_enc, audio1_enc, audio2_enc)
        dist1 = self.d(audio1_enc, video_enc).reshape(-1)
        dist2 = self.d(audio2_enc, video_enc).reshape(-1)
        return loss.mean(), ((dist1 < dist2) * torch.ones(dist1.shape).to(dist1.device)).reshape(-1)
