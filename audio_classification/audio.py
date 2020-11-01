
import random
from functools import partial

import torch
import torchvision
import torchaudio

from torchvision.datasets import DatasetFolder

def audio_loader(path, max_length_in_seconds, pad_and_truncate):
    vframe, aframe, info = torchvision.io.read_video(path)
    sample_rate = info['audio_fps']
    transform = torchvision.transforms.Resize((360, 360))
    #torchaudio.load(path)

    max_length_audio = int(info['audio_fps'] * max_length_in_seconds)
    max_length_video = int(info['video_fps'] * max_length_in_seconds)
    vframe = transform(vframe.permute(0, 3, 1, 2))
    aframe = aframe[0, :max_length_audio]
    vframe = vframe[:50].permute(1, 0, 2, 3)
    return aframe, vframe


def get_audio_dataset(datafolder, max_length_in_seconds=2, pad_and_truncate=False):
    loader_func = partial(
        audio_loader,
        max_length_in_seconds=max_length_in_seconds,
        pad_and_truncate=pad_and_truncate,
    )
    dataset = DatasetFolder(datafolder, loader_func, ".mp4")
    return dataset
