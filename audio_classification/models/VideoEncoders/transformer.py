import torch 
import torch.nn.functional as F
from torch import nn
import numpy as np 
import math 
import torchvision
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pretrainedmodels

class VideoEnc(nn.Module):
    def __init__(self, video_size=(48, 360, 360), out_dim=128, dmodel=512, num_heads=2, num_layers=2,
                 hidden_size=512, dropout=0.5):
        super(VideoEnc, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True, progress=True)
        self.base = nn.Sequential(*list(resnet18.children())[:-3])
        self.encode = nn.Linear(135424, dmodel)
        self.pos_encoder = PositionalEncoding(dmodel, dropout)
        encoder_layers = TransformerEncoderLayer(dmodel, num_heads,
                                                 hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.out = nn.Linear(dmodel * video_size[0], out_dim)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def forward(self, x):
        b = x.size(0)
        t = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x).view(b, t, -1).permute(1, 0, 2)
        x = self.encode(x)
        x = self.pos_encoder(x)
        encoded = torch.relu(self.transformer_encoder(x, self.src_mask))
        return (self.out(encoded.reshape(b, -1)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)