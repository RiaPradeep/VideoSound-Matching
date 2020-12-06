
import random
from functools import partial
import torch.nn.functional as F

import torch
import torchvision
import torchaudio

from torchvision.datasets import DatasetFolder, ImageFolder
from os import listdir
from os.path import isfile, join


def audio_loader(path, max_length_in_seconds, pad_and_truncate):
    max_length_in_seconds = 2
    data_class = path.split("/")[1]
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
    class_nums = {
        "acoustic_guitar": 0,
        "car": 1,
        "cat": 2,
        "male_speech": 3,
        "bark": 4,
        "faucet": 5
    }
    
    dataset = {}
    for c in class_nums:
        dataset[class_nums[c]] = SingleDataset(join(datafolder, c), loader_func)

    return dataset
