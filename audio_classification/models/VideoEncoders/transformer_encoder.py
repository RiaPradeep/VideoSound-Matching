import torch 
import torch.nn.functional as F
from torch import nn
import numpy as np 
import math 
import torchvision
from torch.autograd import Variable



class Semi_Transformer(nn.Module):
    def __init__(self, num_classes, seq_len):
        super(Semi_Transformer, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.tail = Tail(num_classes, seq_len)

    def forward(self, x):
        b = x.size(0)
        print(x.size())
        exit(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        # x: (b,t,2048,7,4)
        return self.tail(x, b , t )