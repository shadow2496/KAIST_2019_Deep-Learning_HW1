import numpy as np
import torch
from torch import nn, optim
import torchvision.models


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
