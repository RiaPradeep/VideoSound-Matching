import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):

    def __init__(self, input_size, num_sources, dmodel, num_heads, num_layers,
                 hidden_size, dropout=0.5):
        super().__init__()
        self.m = input_size
        self.s = num_sources
        self.dmodel = dmodel
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(dmodel, dropout)
        encoder_layers = TransformerEncoderLayer(dmodel, num_heads,
                                                 hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.encoder = nn.Sequential(nn.Linear(2 * input_size, dmodel),
                                    nn.ReLU())

        self.decoder = nn.Sequential(nn.Linear(dmodel, dmodel),
                                    nn.ReLU())
        self.out = nn.Linear(dmodel, out_dim)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        b = x.size(0)
        n = x.size(2)
        x = x.permute(2, 0, 1, 3).reshape(n, b, -1)
        x = self.encoder(x) * math.sqrt(self.dmodel)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, None)
        model_output = self.out(self.decoder(out))
        return model_output

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
