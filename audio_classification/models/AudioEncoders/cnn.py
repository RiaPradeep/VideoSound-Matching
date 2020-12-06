import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class AudioEnc(nn.Module):
    def __init__(self, num_classes=2, audio_size = (257, 690), channel1=2, channel2=32, num_layers=3,
                    kernel_size=(5, 5), padding=(2, 2), stride=(3, 3), out_dim=700):
        super(AudioEnc, self).__init__()
        in_channels = [channel1] + [channel2 * 2 * (i+1) for i in range(num_layers)]
        #[channel1*(2**i) for i in range(num_layers)]

        # channel2, channel3]
        in_size = audio_size
        out_seq_len = in_size
        
        layers = []
        for _ in range(len(in_channels)-1):
            out_seq_len = conv2D_output_size(out_seq_len, padding, kernel_size, stride)
        
        p = (kernel_size[0]-1)//2
        for i in range(len(in_channels) -1):
            layers.append(nn.Sequential(nn.Conv2d(in_channels[i], in_channels[i+1], (kernel_size), (1, 1), (p, p)),
                                            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
                                            nn.BatchNorm2d(in_channels[i+1]),
                                            nn.LeakyReLU(0.1, inplace=False)))
            torch.nn.init.xavier_uniform_(layers[-1][0].weight)
            
        self.audio = nn.ModuleList(layers)
        self.out = nn.Linear(in_channels[-1]*out_seq_len[0] * out_seq_len[1], out_dim)
        torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        b = x.size(0)
        for audio_layer in self.audio:
            x = audio_layer(x)

        x = x.view(b, -1)
        return self.out(x)