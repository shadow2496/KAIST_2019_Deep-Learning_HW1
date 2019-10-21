from torch import nn, optim
import torchvision.models

from utils import load_checkpoints, initialize_weights


class FCN32s(nn.Module):
    def __init__(self, vgg16):
        super(FCN32s, self).__init__()

        vgg16.features[0].padding = (100, 100)
        for i in [4, 9, 16, 23, 30]:
            vgg16.features[i].ceil_mode = True
        self.features = vgg16.features

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
        for i in [0, 3]:
            l1 = vgg16.classifier[i]
            l2 = self.conv_layers[i]
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data)

    def forward(self, x):
        h = self.features(x)
        h = self.conv_layers(h)
        return h[:, :, 19:19 + x.size(2), 19:19 + x.size(3)].contiguous()


class FCN16s(nn.Module):
    def __init__(self, fcn32s):
        super(FCN16s, self).__init__()

        self.features = fcn32s.features

        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, bias=True),
            nn.ReLU(),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Dropout2d(),

            nn.Conv2d(4096, 21, kernel_size=1, bias=True),
            nn.ConvTranspose2d(21, 21, kernel_size=4, stride=2, bias=False)
        )
        self.skip_layer = nn.Conv2d(512, 21, kernel_size=1, bias=True)
        self.upsample_layer = nn.ConvTranspose2d(21, 21, kernel_size=32, stride=16, bias=False)

        self.conv_layers.apply(initialize_weights)
        self.skip_layer.apply(initialize_weights)
        self.upsample_layer.apply(initialize_weights)
        for i in [0, 3, 6]:
            l1 = fcn32s.conv_layers[i]
            l2 = self.conv_layers[i]
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)

    def forward(self, x):
        pool4 = self.features[:24](x)
        h = self.features[24:](pool4)

        conv7 = self.conv_layers(h)
        pool4 = self.skip_layer(pool4)
        h = conv7 + pool4[:, :, 5:5 + conv7.size(2), 5:5 + conv7.size(3)]

        h = self.upsample_layer(h)
        return h[:, :, 27:27 + x.size(2), 27:27 + x.size(3)].contiguous()


class FCN(nn.Module):
    def __init__(self, config):
        super(FCN, self).__init__()

        if config.network == 'fcn32s':
            vgg16 = torchvision.models.vgg16(pretrained=(config.load_iter == 0))
            self.network = FCN32s(vgg16)
        elif config.network == 'fcn16s':
            vgg16 = torchvision.models.vgg16(pretrained=False)
            fcn32s = FCN32s(vgg16)
            if config.load_iter == 0:
                load_checkpoints(fcn32s, 'fcn32s', config.checkpoint_dir, 100000)
            self.network = FCN16s(fcn32s)

        if config.is_train:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
            self.optimizer = optim.SGD(self.network.parameters(), lr=config.lr,
                                       momentum=config.momentum, weight_decay=config.weight_decay)

    def forward(self, x):
        return self.network(x)
