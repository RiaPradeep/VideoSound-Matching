import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import * 
from .AudioEncoders import lstm as acnn

class Model(nn.Module):
    def __init__(self, audio_size = (1, 690, 257),
                    num_classes=2, channel1=2, channel2=64, channel3=128, 
                    kernel_size=(5, 5), padding=(2, 2), stride=(3, 3), out_dim=400):
        super(Model, self).__init__()
        # Audio encoder model
        self.audio_enc = acnn.AudioEnc(audio_size=audio_size[1:], 
                                        out_dim=out_dim)
        # Map to common embedding space
        self.common = nn.Sequential(nn.Linear(out_dim, out_dim))
        # Predict similarity score
        self.predict = nn.Sequential(nn.Linear(2*out_dim, out_dim),
                                    nn.ReLU(),
                                    nn.Linear(out_dim, out_dim//2),
                                    nn.ReLU(),
                                    nn.Linear(out_dim//2, 1),
                                    nn.Sigmoid())
    
    def forward(self, audio1, audio2):
        audio1_enc = self.audio_enc(audio1)
        audio1_out =  (self.common(audio1_enc))
        audio2_enc = self.audio_enc(audio2)
        audio2_out =  (self.common(audio2_enc))
        output = torch.cat((audio1_out, audio2_out), 1).to(audio2_out.device)
        pred = self.predict(output)
        return pred, (audio1_enc, audio2_enc)
