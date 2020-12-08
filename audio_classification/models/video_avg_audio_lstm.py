import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 
from .AudioEncoders import lstm as alstm
from .VideoEncoders import cnn_avg as vcnn_avg

class Model(nn.Module):
    def __init__(self, audio_size = (1, 257, 690), video_size=(1, 48, 360, 360),
                    num_classes=2, channel1=2, channel2=64, channel3=128, loss_type='triplet',
                    kernel_size=(5, 5), padding=(2, 2), stride=(3, 3), out_dim=256):
        super(Model, self).__init__()
        self.video_enc = vcnn_avg.VideoEnc(video_size=video_size[1:], out_dim=out_dim)
        self.audio_enc = alstm.AudioEnc(audio_size=audio_size[1:], out_dim=out_dim)
        self.out = nn.Sequential(nn.Linear(out_dim, out_dim))
        self.linear = nn.Sequential(nn.Linear(2*out_dim, 1),
                                    nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()
        self.loss_type = loss_type

    def forward_bce(self, audio, video):
        audio1_enc = self.audio_enc(audio)
        audio1_out =  self.out(audio1_enc)
        video_enc = self.video_enc(video)
        video_out = self.out(video_enc)
        output = torch.cat((video_out, audio1_out), 1).to(video_enc.device)
        pred = self.linear(output)
        return pred, None
    
    def forward_trip(self, audio, video):
        audio1_enc = self.audio_enc(audio[0])
        audio1_out =  self.sigmoid(self.out(audio1_enc))
        audio2_enc = self.audio_enc(audio[1])
        audio2_out =  self.sigmoid(self.out(audio2_enc))
        video_enc = self.video_enc(video)
        video_out =  self.sigmoid(self.out(video_enc))
        return audio1_enc, audio2_enc, video_out

    def forward_mult(self, audio, video):
        audio1_enc = self.audio_enc(audio[0])
        audio1_out =  self.out(audio1_enc)
        audio2_enc = self.audio_enc(audio[1])
        audio2_out =  self.out(audio2_enc)
        video1_enc = self.video_enc(video[0])
        video1_out =  self.out(video1_enc)
        video2_enc = self.video_enc(video[1])
        video2_out =  self.out(video2_enc)
        output = torch.cat((video1_out, audio1_out), 1).to(video2_enc.device)
        pred = self.linear(output)
        return pred, audio1_enc, audio2_enc, video1_out, video2_out

    def forward_cos(self, audio, video):
        audio1_enc = self.audio_enc(audio)
        audio1_out =  self.out(audio1_enc)
        video_enc = self.video_enc(video)
        video_out = self.out(video_enc)
        output = torch.cat((video_out, audio1_out), 1).to(video_enc.device)
        pred = self.linear(output)
        return pred, (audio1_out, video_out)

    def forward(self, audio, video):
        if self.loss_type == 'bce':
            return self.forward_bce(audio, video)
        elif self.loss_type == 'triplet':
            return self.forward_trip(audio, video)
        elif self.loss_type == 'multi':
            return self.forward_mult(audio, video)
        elif self.loss_type == 'cos':
            return self.forward_cos(audio, video)

