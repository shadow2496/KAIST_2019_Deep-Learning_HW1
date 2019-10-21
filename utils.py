import os

import numpy as np
import torch
from torch import nn


def load_checkpoints(network, name, checkpoint_dir, load_iter):
    filename = '{:06d}-{}.ckpt'.format(load_iter, name)
    print('Loading {}...'.format(filename))
    network_path = os.path.join(checkpoint_dir, filename)
    network.load_state_dict(torch.load(network_path, map_location=lambda storage, loc: storage))


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)


def bitget(byteval, idx):
    return (byteval & (1 << idx)) != 0


def get_colormap(n=256):
    cmap = torch.zeros((n, 3))
    for i in range(n):
        num = i
        r, g, b = 0, 0, 0
        for j in range(8):
            r = np.bitwise_or(r, (bitget(num, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(num, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(num, 2) << 7 - j))
            num = (num >> 3)
        cmap[i, 0] = r / 255
        cmap[i, 1] = g / 255
        cmap[i, 2] = b / 255
    return cmap
