'''not done yet'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class VideoEnc(nn.Module):
    def __init__(self, video_size=(48, 360, 360),
                drop_p=0.2, fc_hidden1=128, out_dim=128, ch1=8, ch2=2, 
                k1=(5, 5), k2=(3, 3), s1=(2, 2), s2=(2, 2), pd1=(0, 0), pd2=(0, 0),
                hidden_size=256, num_layers=8, bidirectional=True):
        super(VideoEnc, self).__init__()
        # set video dimension
        self.t_dim = video_size[0]
        self.img_x = video_size[1]
        self.img_y = video_size[2]
        self.fc_hidden1 = fc_hidden1
        self.drop_p = drop_p
        self.ch1, self.ch2 = ch1, ch2
        self.k1, self.k2 = k1, k2  # 3d kernel size
        self.s1, self.s2 = s1, s2  # 3d strides
        self.pd1, self.pd2 = pd1, pd2  # 3d padding
        self.conv1_outshape = conv2D_output_size(video_size[1:], self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.spatial_enc = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.ch1,  kernel_size=self.k1, stride=self.s1, padding=self.pd1), 
                                        nn.BatchNorm3d(self.ch1),
                                         nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
                                         nn.BatchNorm3d(self.ch2),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(self.drop_p))
        self.audio = nn.LSTM(self.conv2_outshape, hidden_size=hidden_size, num_layers=num_layers, 
                                batch_first=True, bidirectional=bidirectional)
        mult_val = 2 if bidirectional else 1
        self.out = nn.Linear(mult_val * hidden_size * video_size[0], out_dim)

    def forward(self, video):
        b = video.size(0)
        
        spatial = video.reshape(-1, 1, self.img_x, self.img_y)
        x = self.spatial_enc(spatial)
        x = F.relu(x).reshape(b, )
        x = self.video_enc(video).view(b, self.t_dim, -1)

        temp = x.permute(0, 2, 1).reshape(-1, -1, 1)

        x = F.dropout(x, p=self.drop_p, training=self.training)
        return x
