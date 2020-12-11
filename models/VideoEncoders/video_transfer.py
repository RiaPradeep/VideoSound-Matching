# TODO does this work?
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class VideoEnc(nn.Module):
    def __init__(self, v_shape=(48, 360, 360),
                drop_p=0.2, fc_hidden1=128, out_dim=128, ch1=8, ch2=2, 
                k1=(5, 5, 5), k2=(3, 3, 3), s1=(2, 2, 2), s2=(2, 2, 2), pd1=(0, 0, 0), pd2=(0, 0, 0)):
        super(VideoEnc, self).__init__()
        # set video dimension
        self.t_dim = v_shape[0]
        self.img_x = v_shape[1]
        self.img_y = v_shape[2]
        self.resnet = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        self.video_enc = nn.Sequential(
            self.resnet.stem,
            self.resnet.layer1,
            self.resnet.layer2,
            nn.Conv3d(in_channels=128, out_channels=16, kernel_size=k2, stride=s2,
                               padding=pd2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(drop_p),
        )
        self.fc1 = nn.Linear(16 * 5 * 44 * 44, fc_hidden1)  # fully connected hidden layer
        self.drop = nn.Dropout3d(drop_p)

    def forward(self, video):
        b = video.size(0)
        x = self.video_enc(video).view(b, -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        return x
