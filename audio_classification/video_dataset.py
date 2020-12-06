
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

def video_loader(path, max_length_in_seconds, pad_and_truncate):
    # audio works with pts?
    max_length_in_seconds = 2
    data_class = path.split("/")[1]
    vframe, aframe, info = torchvision.io.read_video(path)
    old_sample_rate = info['audio_fps']     
    aframe = aframe[0]
    #waveform, sample_rate = torchaudio.load(path)
    vframe = vframe.permute(0, 3, 1, 2)
    # current shape of vframe is T, C, H, W
    max_length_video = 24 * max_length_in_seconds # average of 24 fps
    vframe = torch.stack(list(resize_video(vframe)), dim=0) # T, C, H, W
    if vframe.shape[0] < max_length_video:
        print("video short -", path)
    # pad if necessery
    diff_video = max(0, max_length_video-vframe.shape[0])
    vframe = torch.cat([vframe, torch.zeros((diff_video, vframe.shape[1], vframe.shape[2], vframe.shape[3]))], dim=0)
    vframe = vframe[:max_length_video].permute(1, 0, 2, 3) # C, T, H, W
    # convert to stft
    # 2, N, T
    return 1, vframe 

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, path, loader_func):
        super(SingleDataset, self).__init__()
        self.all_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        self.loader_func = loader_func 

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        return self.loader_func(self.all_files[idx])

def get_video_dataset(datafolder, max_length_in_seconds=2, pad_and_truncate=False):
    loader_func = partial(
        video_loader,
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
