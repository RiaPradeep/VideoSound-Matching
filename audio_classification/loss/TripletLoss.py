import torch
import torch.nn.functional as F


class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss = torch.nn.TripletMarginLoss()
    
    def forward(self, audio1_enc, audio2_enc, video_enc, label):
        zero_label_loss = self.loss(video_enc, audio1_enc, audio2_enc)
        one_label_loss = self.loss(video_enc, audio2_enc, audio1_enc)
        loss = (1-label) * zero_label_loss + label * one_label_loss
        dist1 = F.pairwise_distance(audio1_enc, video_enc, keepdim=True)
        dist2 = F.pairwise_distance(audio2_enc, video_enc, keepdim=True)
        distances = torch.cat([dist1, dist2], dim=1)
        return loss, distances
