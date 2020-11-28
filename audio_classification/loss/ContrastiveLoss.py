import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Find the pairwise distance or eucledian distance of two output feature vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return euclidean_distance.reshape(-1, 1), loss_contrastive

class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss = ContrastiveLoss()
    
    def forward(self, audio1_enc, audio2_enc, video_enc, label):
        loss1 = self.loss(audio1_enc, video_enc, label)
        loss2 = self.loss(audio2_enc, video_enc, 1-label)
        distances = torch.cat([loss1[0], loss2[0]], dim=1)
        return loss1[1] + loss2[1], distances