'''
    TODO check if lstm can accept this as spatial input
    not done yet
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class VideoEnc(nn.Module):
    def __init__(self, video_size=(48, 360, 360),
                drop_p=0.2, hidden_size=8, out_dim=128, num_layers=2, 
                bidirectional=True, kernel_size=(5, 5),  stride=(3, 3), padding=(1, 1),
                channel1=3):
        super(VideoEnc, self).__init__()
        in_channels = [channel1] + [8 * ((i+1)) for i in range(num_layers)]
        self.t_dim = video_size[0]
        self.img_x = video_size[1]
        self.img_y = video_size[2]
        
        layers = []
        out_seq_len = video_size[1:]

        for _ in range(len(in_channels)-1):
            out_seq_len = conv2D_output_size(out_seq_len, padding, kernel_size, stride)
        
        p = (kernel_size[0]-1)//2
        
        for i in range(len(in_channels) -1):
            layers.append(nn.Sequential(nn.Conv2d(in_channels[i], in_channels[i+1], (kernel_size), (1, 1), (p, p)),
                                            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
                                            nn.BatchNorm2d(in_channels[i+1]),
                                            nn.LeakyReLU(0.2, inplace=False)))
            torch.nn.init.xavier_uniform_(layers[-1][0].weight)

            
        self.spatial_enc = nn.ModuleList(layers)
        self.out = nn.Linear(in_channels[-1]*out_seq_len[0] * out_seq_len[1], out_dim)
        torch.nn.init.xavier_uniform_(self.out.weight)

        self.drop_p = drop_p

    def forward(self, video):
        b = video.size(0)
        t = video.size(2)
        # b, 3, 48, 360, 360=> b, 48, 3, 360, 360 
        spatial = video.permute(0, 2, 1, 3, 4).reshape(-1, 3, self.img_x, self.img_y)
        x = spatial
        for spatial_layer in self.spatial_enc:
            x = spatial_layer(x)
        # b, C, H, W
        x = torch.mean(x.reshape(b, t, -1), dim=1)
        # b, C*H*W
        x = x.reshape(b, -1)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        return self.out(x)
