import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class AudioEnc(nn.Module):
    def __init__(self, num_classes=2, audio_size = (690, 257), 
                    dmodel=200, hidden_size=200, out_dim=128,
                    num_layers=3, bidirectional=False):
        super(AudioEnc, self).__init__()
        self.in_size = audio_size
        self.enc = nn.Linear(2*audio_size[1], dmodel)
        self.audio = nn.LSTM(dmodel, hidden_size=hidden_size, num_layers=num_layers, 
                                batch_first=True, bidirectional=bidirectional)
        mult_val = 2 if bidirectional else 1
        self.linear = nn.Linear(mult_val * hidden_size, out_dim)

    #TODO check x shape- this assumes third is num time frames
    # x is currently b, 2, T, N
    def forward(self, x):
        b = x.size(0)
        # x: b, T, 2, N
        x = x.permute(0, 2, 1, 3).reshape(b, self.in_size[0], 2*self.in_size[1])
        audio_enc = F.relu(self.enc(x))
        audio_enc, _ = self.audio(audio_enc)
        audio_enc = audio_enc[:, -1]
        audio_enc = self.linear(audio_enc.reshape(b, -1))
        return audio_enc