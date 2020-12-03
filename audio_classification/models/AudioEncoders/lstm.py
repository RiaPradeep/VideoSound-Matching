import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class AudioEnc(nn.Module):
    def __init__(self, num_classes=2, audio_size = (257, 690), 
                    dmodel=256, hidden_size=64, out_dim=128,
                    num_layers=3, bidirectional=True):
        super(AudioEnc, self).__init__()
        self.in_size = audio_size
        self.enc = nn.Linear(2*audio_size[1], dmodel)
        self.audio = nn.LSTM(dmodel, hidden_size=hidden_size, num_layers=num_layers, 
                                batch_first=True, bidirectional=bidirectional)
        mult_val = 2 if bidirectional else 1
        self.linear = nn.Linear(mult_val * hidden_size * audio_size[0], out_dim)

    #TODO check x shape- this assumes third is num time frames
    def forward(self, x):
        b = x.size(0)
        x = x.permute(0, 2, 1, 3).reshape(b, self.in_size[0], 2*self.in_size[1])
        audio_enc = self.enc(x)
        audio_enc, _ = self.audio(audio_enc)
        audio_enc = self.linear(audio_enc.view(b, -1))
        return audio_enc