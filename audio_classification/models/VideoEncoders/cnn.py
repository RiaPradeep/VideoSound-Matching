import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 

class VideoEnc(nn.Module):
    def __init__(self, video_size=(48, 360, 360),
                drop_p=0.2, fc_hidden1=128, out_dim=128, num_layers=6, 
                kernel_size=(5, 5, 5),  stride=(1, 2, 2), padding=(0, 0, 0),
                channel1=3):
        super(VideoEnc, self).__init__()
        # set video dimension
        self.t_dim = video_size[0]
        self.img_x = video_size[1]
        self.img_y = video_size[2]
        self.fc_hidden1 = fc_hidden1
        self.drop_p = drop_p
        p = (kernel_size[0]-stride[0])//2
        padding = (p, p, p)
        in_channels = [channel1*(2**i) for i in range(num_layers)]
        in_channels += [channel1*(2**i) for i in range(num_layers, num_layers+4)]
        layers = []
        out_shape = video_size
        for _ in in_channels:
            out_shape = conv3D_output_size(out_shape, padding, kernel_size, stride)
            print(out_shape)
        
        for _ in range(4):
            out_shape = conv3D_output_size(out_shape, padding, kernel_size, (2, 1, 1))
            print(out_shape)

        for i in range(len(in_channels) -1):
            layers.append(nn.Sequential(nn.Conv3d(in_channels[i], in_channels[i+1], (kernel_size), padding),
                                            nn.MaxPool3d(kernel_size, stride=stride),
                                            nn.BatchNorm2d(in_channels[1]),
                                            nn.LeakyReLU(0.2, inplace=True),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout3d(self.drop_p)))
        for i in range(4):
            layers.append(nn.Sequential(nn.Conv3d(in_channels[i], in_channels[i+1], (kernel_size), padding),
                                            nn.MaxPool3d(kernel_size, stride=(2, 1, 1)),
                                            nn.BatchNorm2d(in_channels[1]),
                                            nn.LeakyReLU(0.2, inplace=True),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout3d(self.drop_p)))

        self.video_enc = nn.ModuleList(layers)
        self.fc1 = nn.Linear(in_channels[-1] * out_shape[0] * out_shape[1] * out_shape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, out_dim)

    def forward(self, video):
        b = video.size(0)
        print("VIDEO", video.size())
        video = x
        for v_layer in self.video_enc:
            x = v_layer(x)
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        return x
