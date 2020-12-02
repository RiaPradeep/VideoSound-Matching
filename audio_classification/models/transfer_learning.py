import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape
    
def conv1d_output_size(seq_len, padding, kernel_size, stride):
    outshape = (np.floor((seq_len + 2 * padding - (kernel_size - 1) - 1) / stride + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=24, img_x=360, img_y=360, drop_p=0.2, fc_hidden1=128, fc_hidden2=128, channel1=8, channels=2, 
                    k1=(5, 5, 5), k2=(3, 3, 3), s1=(2, 2, 2), s2=(2, 2, 2), pd1=(0, 0, 0), pd2=(0, 0, 0)):
        super(CNN3D, self).__init__()
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
        # self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.drop = nn.Dropout3d(drop_p)


        # print(self.video_enc)
        # # set video dimension
        # self.t_dim = t_dim
        # self.img_x = img_x
        # self.img_y = img_y
        # # fully connected layer hidden nodes
        # self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        # self.drop_p = drop_p
        # self.ch1, self.ch2 = 8, 16
        # self.k1, self.k2 = k1, k2  # 3d kernel size
        # self.s1, self.s2 = s1, s2  # 3d strides
        # self.pd1, self.pd2 = pd1, pd2  # 3d padding

        # # compute conv1 & conv2 output shape
        # self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        # self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        # self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
        #                        padding=self.pd1)
        # self.bn1 = nn.BatchNorm3d(self.ch1)
        # self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
        #                        padding=self.pd2)
        # self.bn2 = nn.BatchNorm3d(self.ch2)
        # self.relu = nn.ReLU(inplace=True)
        # self.drop = nn.Dropout3d(self.drop_p)
        # self.pool = nn.MaxPool3d(2)
        # self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
        #                      self.fc_hidden1)  # fully connected hidden layer
        # self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x_3d):
        x = self.video_enc(x_3d)
        # # Conv 1
        # x = self.conv1(x_3d)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # # Conv 2
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.drop(x)
        # x = F.dropout(x, p=drop_p, training=self.training)
        print(x.shape)
        return x

        # return x

class AudioCNN(nn.Module):
    def __init__(self, num_classes=2, in_size = 44100, channel1=1, channel2=64, channel3=128, 
                    kernel_size=80, padding=2, stride=8):
        super(AudioCNN, self).__init__()
        self.video_enc = CNN3D()
        in_channels = [channel1, channel2, channel3]
        out_seq_len = in_size
        for i in in_channels:
            out_seq_len = conv1d_output_size(out_seq_len, padding, kernel_size, stride)

        self.audio = nn.Sequential(
            nn.Conv1d(channel1, channel2, kernel_size, stride, padding),
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel2, channel3, kernel_size, stride, padding),
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel3, channel2, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Linear(out_seq_len * channel2, 128)

    def forward(self, audio1, audio2, video):
        b = audio1.shape[0]
        audio1_enc = self.linear(self.audio(audio1.unsqueeze(1)).view(b, -1))
        audio2_enc = self.linear(self.audio(audio2.unsqueeze(1)).view(b, -1))
        video_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        video_enc = self.video_enc(video.type(video_type)).view(b, -1)
        return (audio1_enc, audio2_enc, video_enc)
