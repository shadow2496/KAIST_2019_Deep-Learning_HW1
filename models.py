from torch import nn, optim
import torchvision.models

from utils import initialize_weights


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
        self.network = FCN32s(vgg16.features)

        if self.is_train:
            self.optimizer = optim.SGD(self.network.parameters(), lr=config.lr,
                                       momentum=config.momentum, weight_decay=config.weight_decay)

    def forward(self, x):
        return self.network(x)
