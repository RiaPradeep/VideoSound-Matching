import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AudioEnc(nn.Module):
    def __init__(self, audio_size, out_dim=128, dmodel=128, num_heads=2, num_layers=2,
                 hidden_size=256, dropout=0.5):
        super().__init__()
        print(audio_size)
        self.m = audio_size[-1]
        self.n = audio_size[0]
        self.dmodel = dmodel
        self.model_type = 'Transformer'
        self.src_mask = None
        self.encoder = nn.Linear(2*self.m, dmodel )
        self.pos_encoder = PositionalEncoding(dmodel, dropout)
        encoder_layers = TransformerEncoderLayer(dmodel, num_heads,
                                                 hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.out = nn.Linear(dmodel * self.n, out_dim)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        b = x.size(0)
        n = x.size(2)
        # b, 2, n, m => n, b, 2, m => n, b, 2m
        x = x.permute(2, 0, 1, 3).reshape(n, b, -1)
        x = torch.relu(self.encoder(x)) * math.sqrt(self.dmodel)
        x = self.pos_encoder(x)
        # n, b, 
        encoded = torch.relu(self.transformer_encoder(x, self.src_mask))
        return self.out(encoded.reshape(b, -1))

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