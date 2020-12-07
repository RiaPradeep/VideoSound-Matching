
import random
from functools import partial
import torch.nn.functional as F

import torch
import torchvision
import torchaudio

from torchvision.datasets import DatasetFolder, ImageFolder
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import librosa.display

import numpy as np
import pandas as pd
import librosa
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def display(spect, sample_rate, y_axis='mel',x_axis='time', i="0"):
    plt.figure(figsize=(12, 4))
    ax = plt.axes()
    ax.set_axis_off()
    plt.set_cmap('hot')    
    db_data = librosa.power_to_db(np.abs(spect)**2, ref=np.max)
    librosa.display.specshow(db_data, sr=sample_rate, y_axis=y_axis, x_axis=x_axis)
    plt.colorbar()
    plt.savefig(i+"out.png", bbox_inches='tight', transparent=True, pad_inches=0.0 )
    plt.close()

def create_composite(tensor):
    real = tensor[0].numpy().T
    imag = tensor[1].numpy().T
    composite = real + 1j * imag
    return composite


def visualize_input(tensor, sample_rate, i):
    composite = create_composite(tensor)
    fig = plt.figure(figsize=(3, 1))
    plt.subplot(1, 1, 1)
    display(composite, sample_rate=sample_rate,i=i)

def audio_loader(path, max_length_in_seconds, pad_and_truncate):
    max_length_in_seconds = 2
    data_class = path.split("/")[-2]
    max_length_in_seconds=4
    vframe, aframe, info = torchvision.io.read_video(path)
    old_sample_rate = info['audio_fps']

  
    sample_rate = 8000
    aframe = torchaudio.transforms.Resample(old_sample_rate, sample_rate)(aframe)

    #waveform, sample_rate = torchaudio.load(path)
    aframe = aframe[0]
    # current shape of vframe is T, C, H, W
    max_length_audio = int(sample_rate * max_length_in_seconds)
    if aframe.shape[0] < max_length_audio:
        print("audio short -", path)

    diff_audio = max(0, (max_length_audio-aframe.shape[0]))
    aframe = torch.cat([aframe, torch.zeros((diff_audio))], dim=0)
    aframe = aframe[:max_length_audio]
    # convert to stft
    # 2, N, T
    aframe = torch.stft(aframe, n_fft=512, return_complex=False).permute(2, 1, 0)
    #visualize_input(aframe, sample_rate, data_class + path.split("/")[-1])
    #exit(0)
    return aframe, vframe


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, path, loader_func):
        super(SingleDataset, self).__init__()
        self.all_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        self.loader_func = loader_func 

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        return self.loader_func(self.all_files[idx])

def get_audio_dataset(datafolder, max_length_in_seconds=2, pad_and_truncate=False):
    loader_func = partial(
        audio_loader,
        max_length_in_seconds=max_length_in_seconds,
        pad_and_truncate=pad_and_truncate,
    )
    
    dataset_idx = {}
    class_nums = ["acoustic_guitar", "cowbell", "knock", 
                    "applause", "duck", 
                    "tearing", 
                    "telephone_bell_ring", 
                    "male_speech", "bark", 
                    "typing", "faucet", "piano", 
                    "bird", "vacuum_cleaner", "rain",
                    "water", "raindrop", "saxophone", "writing"]
    
    dataset = {}
    i = 0
    for c in class_nums:
        d = SingleDataset(join(datafolder, c), loader_func)
        if len(d) >= 99:
            dataset[i] = d
            print(c)
            i += 1
        if i==15:
            break 
    return dataset
