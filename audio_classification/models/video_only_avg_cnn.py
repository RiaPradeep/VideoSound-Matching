import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 
from .AudioEncoders import cnn as acnn
from .VideoEncoders import cnn_avg as vcnn_avg

class Model(nn.Module):
    def __init__(self, audio_size = (1, 257, 690), video_size=(1, 48, 360, 360),
                    num_classes=2, channel1=2, channel2=64, channel3=128, 
                    kernel_size=(5, 5), padding=(2, 2), stride=(3, 3), out_dim=128):
        super(Model, self).__init__()

        self.video_enc = vcnn_avg.VideoEnc(video_size=video_size[1:], out_dim=out_dim)
        self.out = nn.Sequential(nn.Linear(out_dim, out_dim))

    def forward(self, video1, video2):
        video1_enc = self.video_enc(video1)
        video1_out = self.out(video1_enc)
        video2_enc = self.video_enc(video2)
        video2_out = self.out(video2_enc)
        return video1_out, video2_out
