import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 
from .AudioEncoders import lstm as alstm
from .VideoEncoders import cnn_lstm as vlstm

class Model(nn.Module):
    def __init__(self, audio_size = (1, 257, 690), video_size=(1, 48, 360, 360),
                    num_classes=2, channel1=2, channel2=64, channel3=128, 
                    kernel_size=(5, 5), padding=(2, 2), stride=(3, 3), out_dim=128,
                    loss_type='bce'):
        super(Model, self).__init__()
        # Independent audio and video encoder models
        self.video_enc = vlstm.VideoEnc(video_size=video_size[1:], out_dim=out_dim)
        self.audio_enc = alstm.AudioEnc(audio_size=audio_size[1:], out_dim=out_dim)
        # Map to common embedding space
        self.common = nn.Sequential(nn.ReLU(), 
                                 nn.Linear(out_dim, out_dim),
                                 nn.ReLU(),
                                 nn.Linear(out_dim, out_dim))
        # Predict similarity score
        self.predict = nn.Sequential(nn.Linear(2*out_dim, out_dim),
                                    nn.ReLU(),
                                    nn.Linear(out_dim, out_dim//2),
                                    nn.ReLU(),
                                    nn.Linear(out_dim//2, 1),
                                    nn.Sigmoid())
        self.loss_type = loss_type

    def forward_bce(self, audio, video):
        audio_enc = self.audio_enc(audio)
        audio_out =  self.common(audio_enc)
        video_enc = self.video_enc(video)
        video_out = self.common(video_enc)
        output = torch.cat((video_out, audio_out), 1).to(video_enc.device)
        pred = self.predict(output)
        return pred, None
    
    def forward_trip(self, audio, video):
        audio1_enc = self.audio_enc(audio[0])
        audio1_out =  self.sigmoid(self.common(audio1_enc))
        audio2_enc = self.audio_enc(audio[1])
        audio2_out =  self.sigmoid(self.common(audio2_enc))
        video_enc = self.video_enc(video)
        video_out =  self.sigmoid(self.common(video_enc))
        return audio1_out, audio2_out, video_out

    def forward_mult(self, audio, video):
        audio1_enc = self.audio_enc(audio[0])
        audio1_out =  self.common(audio1_enc)
        audio2_enc = self.audio_enc(audio[1])
        audio2_out =  self.common(audio2_enc)
        video1_enc = self.video_enc(video[0])
        video1_out =  self.common(video1_enc)
        video2_enc = self.video_enc(video[1])
        video2_out =  self.common(video2_enc)
        return audio1_out, audio2_out, video1_out, video2_out

    def forward_cos(self, audio, video):
        audio_enc = self.audio_enc(audio)
        audio_out =  self.common(audio_enc)
        video_enc = self.video_enc(video)
        video_out = self.common(video_enc)
        output = torch.cat((video_out, audio_out), 1).to(video_enc.device)
        pred = self.predict(output)
        return pred, (audio_out, video_out)

    # Call appropriate forward function based on loss type
    def forward(self, audio, video):
        if self.loss_type == 'bce':
            return self.forward_bce(audio, video)
        elif self.loss_type == 'triplet':
            return self.forward_trip(audio, video)
        elif self.loss_type == 'multi':
            return self.forward_mult(audio, video)
        elif self.loss_type == 'cos':
            return self.forward_cos(audio, video)