import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 
from .AudioEncoders import cnn as acnn
from .VideoEncoders import cnn as vcnn

class Model(nn.Module):
    def __init__(self, audio_size = (1, 257, 690), video_size=(1, 48, 360, 360),
                    num_classes=2, channel1=2, channel2=64, channel3=128, 
                    kernel_size=(5, 5), padding=(2, 2), stride=(3, 3), out_dim=128):
        super(Model, self).__init__()
        self.video_enc = vcnn.VideoEnc(video_size=video_size[1:], out_dim=128)
        self.audio_enc = acnn.AudioEnc(audio_size=audio_size[1:], out_dim=128)
        self.out = nn.Linear(out_dim, out_dim)

    def forward(self, audio1, audio2, video):
        b = audio1.shape[0]

        audio1_enc = self.audio_enc(audio1)
        audio2_enc = self.audio_enc(audio2)
        video_enc = self.video_enc(video)
        video_out = self.out(video_enc)
        audio1_out = self.out(audio1_enc)
        audio2_out = self.out(audio2_enc)
        return audio1_out, audio2_out, video_out
