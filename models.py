import numpy as np
import torch
from torch import nn, optim
import torchvision.models


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


class FCN32s(nn.Module):
    def __init__(self, features):
        super(FCN32s, self).__init__()

        self.features = features

        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, bias=True),
            nn.ReLU(),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Dropout2d(),

            nn.Conv2d(4096, 21, kernel_size=1, bias=True),
            nn.ConvTranspose2d(21, 21, kernel_size=64, stride=32, bias=False)
        )

        self.conv_layers.apply(initialize_weights)

    def forward(self, x):
        h = self.features(x)
        h = self.conv_layers(h)
        return h[:, :, 19:19 + x.size(2), 19:19 + x.size(3)].contiguous()


class FCN(nn.Module):
    def __init__(self, config):
        super(FCN, self).__init__()

        self.is_train = config.is_train
        vgg16 = torchvision.models.vgg16(pretrained=self.is_train)
        for param in vgg16.features.parameters():
            param.required_grad = False
        self.network = FCN32s(vgg16.features)

        if self.is_train:
            pass

    def forward(self, x):
        return self.network(x)
