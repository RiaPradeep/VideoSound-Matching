import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class AudioEnc(nn.Module):
    def __init__(self, num_classes=2, audio_size = (257, 690), channel1=2, channel2=64, channel3=128, 
                    kernel_size=(5, 5), padding=(2, 2), stride=(3, 3), out_dim=128):
        super(AudioEnc, self).__init__()
        in_channels = [channel1, channel2, channel3]
        in_size = audio_size
        out_seq_len = in_size
        
        for _ in in_channels:
            out_seq_len = conv2D_output_size(out_seq_len, padding, kernel_size, stride)

        self.audio = nn.Sequential(
            nn.Conv2d(channel1, channel2, (kernel_size), stride, padding),
            nn.BatchNorm2d(channel2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel2, channel3, (kernel_size), stride, padding),
            nn.BatchNorm2d(channel3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel3, channel2, (kernel_size), stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Linear(channel2*out_seq_len[0] * out_seq_len[1], out_dim)

    def forward(self, x):
        b = x.size(0)
        print(x.shape)
        x = self.audio(x).view(b, -1)
        return self.out(x)