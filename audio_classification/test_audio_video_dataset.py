import random
from functools import partial
import torch.nn.functional as F

import torch
import torchvision
import torchaudio

from torchvision.datasets import DatasetFolder, ImageFolder
from os import listdir
from os.path import isfile, join

def resize_video(video, size=(360, 360), interpolation=2):
    transform = torchvision.transforms.Resize(size=size)
    for frame in video:
        yield transform(frame)

def audio_video_loader(path, max_length_in_seconds, pad_and_truncate):
    # audio works with pts?
    data_class = path.split("/")[1]
    vframe, aframe, info = torchvision.io.read_video(path, start_pts=0, end_pts=47500 * max_length_in_seconds, pts_unit="pts")
    if 'audio_fps' not in info:
        print(path)
        exit(0)
        pass
    old_sample_rate = info['audio_fps'] 
    sample_rate = old_sample_rate
    aframe = aframe[0]
    vframe = vframe.permute(0, 3, 1, 2)
    # current shape of vframe is T, C, H, W
    max_length_audio = int(sample_rate * max_length_in_seconds)
    max_length_video = 24 * max_length_in_seconds # average of 24 fps
    vframe = torch.stack(list(resize_video(vframe)), dim=0) # T, C, H, W
    if aframe.shape[0] < max_length_audio:
        print("audio short -", path)
    if vframe.shape[0] < max_length_video:
        print("video short -", path)
    # pad if necessery
    diff_video = max(0, max_length_video-vframe.shape[0])
    diff_audio = max(0, (max_length_audio-aframe.shape[0]))
    vframe = torch.cat([vframe, torch.zeros((diff_video, vframe.shape[1], vframe.shape[2], vframe.shape[3]))], dim=0)
    aframe = torch.cat([aframe, torch.zeros((diff_audio))], dim=0)
    vframe = vframe[:max_length_video].permute(1, 0, 2, 3) # C, T, H, W
    aframe = aframe[:max_length_audio]
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

def get_audio_video_dataset(datafolder, max_length_in_seconds=2, pad_and_truncate=False):
    loader_func = partial(
        audio_video_loader,
        max_length_in_seconds=max_length_in_seconds,
        pad_and_truncate=pad_and_truncate,
    )
    
    dataset_idx = {}
    # ["cat", "engine", "saxophone"]
    #  class_nums = ["acoustic_guitar", "bird", "child_speech", "flute", "piano"]
    # ["acoustic_guitar", "bird", "child_speech", "flute", "piano"]
    class_nums = ["acoustic_guitar", "bird", "child_speech", "flute", "piano"]
    dataset = {}
    i = 0
    for c in class_nums:
        
        d = SingleDataset(join(datafolder, c), loader_func)
        if not(len(d)>= 99):
            print(c)
            exit(0)
        dataset[i] = d
        print(len(d), c)
        i += 1
    return dataset

