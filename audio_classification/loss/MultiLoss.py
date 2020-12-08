import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoMatchingLoss(torch.nn.Module):
    def __init__(self):
        super(VideoMatchingLoss, self).__init__()
        self.loss_class =  nn.BCELoss()
        self.cos = nn.CosineSimilarity()
    
    def forward(self, pred, audio1_enc, audio2_enc, video1_enc, video2_enc, label, reg=0.03):
        loss_1 = self.loss_class(pred, label.view(-1, 1).type(torch.float32)) 
        audio = (audio1_enc, audio2_enc)
        video = (video1_enc, video2_enc)
        cosine_audio_cross = self.cos(audio[0], audio[1])
        cosine_video_cross = self.cos(video[0], video[1])

        cosine_1 = self.cos(audio[0], video[0])
        cosine_2 = self.cos(audio[1], video[1])

        loss_cosine_sim = (1-cosine_1) + (1-cosine_2) 
        loss_cosine_audio_cross_pos = label * (1-cosine_audio_cross + 1 - cosine_video_cross)
        
        cosine_audio_neg = torch.relu(cosine_audio_cross-0.2)
        cosine_video_neg = torch.relu(cosine_video_cross-0.2)
        
        loss_cosine_audio_cross_neg = (1-label) * (cosine_audio_neg + cosine_video_neg)

        loss_2 = loss_cosine_audio_cross_neg + loss_cosine_audio_cross_pos + loss_cosine_sim
        loss = loss_1 + reg * loss_2
        print(torch.min(pred), torch.max(pred)) 
        return loss.mean(), pred.reshape(-1)
