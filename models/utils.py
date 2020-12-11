import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Compute output shape of conv1D
def conv1d_output_size(seq_len, padding, kernel_size, stride):
    outshape = (np.floor((seq_len + 2 * padding - (kernel_size - 1) - 1) / stride + 1).astype(int))
    return outshape

# Compute output shape of conv2D
def conv2D_output_size(img_size, padding, kernel_size, stride):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

# Compute output shape of conv3D
def conv3D_output_size(img_size, padding, kernel_size, stride):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape