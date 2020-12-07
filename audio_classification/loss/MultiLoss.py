import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoMatchingLoss(torch.nn.Module):
    def __init__(self, margin=128):
        super(VideoMatchingLoss, self).__init__()
        self.d = nn.PairwiseDistance(p=2)
        self.loss = torch.nn.TripletMarginLoss(margin=1.5*margin)
        self.margin = margin

    def forward(self, audio1_enc, audio2_enc, video1_enc, video2_enc, reg=0.1):
        t_loss_1 = self.loss(video1_enc, audio1_enc, audio2_enc)
        #t_loss_2 = self.loss(video2_enc, audio2_enc, audio1_enc)
        loss_audio = self.d(audio1_enc, audio2_enc).mean()
        loss_video = self.d(video1_enc, video2_enc).mean()
        loss = t_loss_1 - torch.clip(reg * (loss_audio + loss_video), max=self.margin//2)
        dist1 = self.d(audio1_enc, video1_enc).reshape(-1, 1)
        dist2 = self.d(audio2_enc, video1_enc).reshape(-1, 1)
        #print(loss_audio, loss_video, t_loss_1, t_loss_2)
        return loss.mean(), ((dist1 < dist2) * torch.ones(dist1.shape).to(audio1_enc.device)).reshape(-1)
