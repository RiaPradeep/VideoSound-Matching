'''
    TODO check if lstm can accept this as spatial input
    not done yet
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 
import torchvision

class VideoEnc(nn.Module):
    def __init__(self, video_size=(48, 360, 360),
                drop_p=0.2, hidden_size=8, out_dim=128, num_layers=5, 
                bidirectional=True, kernel_size=(5, 5),  stride=(3, 3), padding=(0, 0),
                channel1=3, dmodel=512):
        super(VideoEnc, self).__init__()
        self.t_dim = video_size[0]
        self.img_x = video_size[1]
        self.img_y = video_size[2]
        
        layers = []
        out_seq_len = video_size[1:]
        resnet18 = torchvision.models.resnet18(pretrained=True, progress=True)
        self.base = nn.Sequential(*list(resnet18.children())[:-3])
        self.encode = nn.Linear(135424, dmodel)
        self.temp_enc = nn.LSTM(dmodel, hidden_size=hidden_size, num_layers=num_layers, 
                                batch_first=True, bidirectional=bidirectional)
        mult_val = 2 if bidirectional else 1
        self.out = nn.Linear(mult_val * hidden_size * video_size[0], out_dim)
        self.drop_p = drop_p

    def forward(self, x):
        b = x.size(0)
        t = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b*t, x.size(2), x.size(3), x.size(4))        
        x = self.encode(torch.relu(self.base(x).view(b, t, -1)))
        x = F.relu(x)
        print(x.size())
        x, _ = self.temp_enc(x)
        x = x.reshape(b, -1)

        x = F.dropout(x, p=self.drop_p, training=self.training)

        return self.out(x)
