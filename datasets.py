import numpy as np
import torch
from torchvision import datasets


class VOCDataset(datasets.VOCSegmentation):
    def __getitem__(self, index):
        img, target = super(VOCDataset, self).__getitem__(index)
        target = np.asarray(target)
        target = torch.as_tensor(target, dtype=torch.long)

        return img, target
