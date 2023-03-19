import random
import numpy as np
import torchvision.datasets as datasets
from PIL import Image


class NoisyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, noise, mode, train=True, transform=None, download=False):
        super(NoisyCIFAR10, self).__init__(root=root, train=train, transform=transform, download=download)
        print(f'Add {noise}% {mode} noise')
        self.targets = np.load(f'label/{mode}/cifar10_{noise}.npy').astype(np.int64)


class NoisyCIFAR100(datasets.CIFAR100):
    def __init__(self, root, noise, mode, train=True, transform=None, download=False):
        super(NoisyCIFAR100, self).__init__(root=root, train=train, transform=transform, download=download)
        print(f'Add {noise}% {mode} noise')
        self.targets = np.load(f'label/{mode}/cifar100_{noise}.npy').astype(np.int64)

