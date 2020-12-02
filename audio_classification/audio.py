
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

def audio_loader(path, max_length_in_seconds, pad_and_truncate):
    # audio works with pts?
    max_length_in_seconds = 2

    data_class = path.split("/")[1]

    vframe, aframe, info = torchvision.io.read_video(path, start_pts=0, end_pts=47500 * max_length_in_seconds, pts_unit="pts")
    aframe = aframe[0]
    #
    
    vframe = vframe.permute(0, 3, 1, 2)
    # current shape of vframe is T, C, H, W
    print(vframe.shape)
    max_length_audio = int(info['audio_fps'] * max_length_in_seconds)
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
    # convert to stft
    aframe = torch.stft(aframe, n_fft=512).permute(2, 0, 1)
    torchvision.io.write_video("VV.mp4", vframe.permute(1, 2, 3, 0), fps=info['video_fps'])
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
        "bird": 1,
        "cat": 2,
        "child_speech": 3,
        "flute": 4,
        "piano": 5,
        "waterfall": 6
    }
    
    dataset = {}
    for c in class_nums:
        dataset[class_nums[c]] = SingleDataset(join(datafolder, c), loader_func)

    return dataset
