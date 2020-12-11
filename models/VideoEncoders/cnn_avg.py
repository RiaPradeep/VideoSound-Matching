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
        in_channels = [channel1] + [16 * ((i+1)) for i in range(num_layers)]
        self.t_dim = video_size[0]
        self.img_x = video_size[1]
        self.img_y = video_size[2]
        layers = []
        resnet18 = torchvision.models.resnet18(pretrained=True, progress=True)
        self.base = nn.Sequential(*list(resnet18.children())[:-3])
        self.encode = nn.Linear(135424, dmodel)
        self.out = nn.Linear(dmodel, out_dim)
        torch.nn.init.xavier_uniform_(self.out.weight)
        self.drop_p = drop_p

    def forward(self, video):
        b = video.size(0)
        t = video.size(2)
        spatial = video.permute(0, 2, 1, 3, 4).reshape(-1, 3, self.img_x, self.img_y)
        x = torch.relu(self.base(spatial).reshape(b, t, -1))
        x = self.encode(x)
        x = torch.mean(x, dim=1)
        x = x.reshape(b, -1)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        return self.out(x)
