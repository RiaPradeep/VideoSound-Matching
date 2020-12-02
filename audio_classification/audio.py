
import random
from functools import partial
import torch.nn.functional as F

import torch
import torchvision
import torchaudio

from torchvision.datasets import DatasetFolder, ImageFolder

def audio_loader(path, max_length_in_seconds, pad_and_truncate):
    # audio works with pts?
    data_class = path.split("/")[1]
    vframe, aframe, info = torchvision.io.read_video(path, start_pts=0, end_pts=47500 * max_length_in_seconds, pts_unit="pts")
    aframe = aframe[0]
    transform = torchvision.transforms.Resize((360, 360))

    max_length_audio = int(info['audio_fps'] * max_length_in_seconds)
    max_length_video = 24 * max_length_in_seconds # average of 24 fps
    vframe = transform(vframe.permute(0, 3, 1, 2)) # T, C, H, W

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
    return aframe, vframe


def get_audio_dataset(datafolder, max_length_in_seconds=2, pad_and_truncate=False):
    loader_func = partial(
        audio_loader,
        max_length_in_seconds=max_length_in_seconds,
        pad_and_truncate=pad_and_truncate,
    )
    
    dataset = {}
    class_nums = {
        "acoustic_guitar": 0,
        "bird": 1,
        "cat": 2,
        "child_speech": 3,
        "flute": 4,
        "piano": 5,
        "waterfall": 6
    }
    
    
    for c in class_nums:
        print(c)
        dataset[class_nums[c]] = (DatasetFolder(datafolder + "/" + c, loader_func, ".mp4"))
    #dataset_sorted = sorted(dataset, lambda x : x[2])
    
    return dataset
