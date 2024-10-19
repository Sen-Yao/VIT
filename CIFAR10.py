# create a dataset for CIFAR10

import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, root: str, oversampling: bool=False, transform=None):
        self.root = root
        self.oversampling = oversampling
        self.transform = transform
        self.data = []
        self.targets = []
        self.classes = range(10)

        self.classes_num = [len(os.listdir(os.path.join(self.root, str(i)))) for i in self.classes]
        self.max_num = max(self.classes_num)

        for i in self.classes:
            diff = self.max_num - self.classes_num[i]
            class_root = os.path.join(self.root, str(i))
            class_data = [os.path.join(class_root, f) for f in os.listdir(class_root)]
            class_one_hot_targets = np.zeros(10)
            class_one_hot_targets[i] = 1
            class_targets = [class_one_hot_targets] * len(class_data)
            if self.oversampling:
                # random sampling
                os_class_data = np.random.choice(class_data, diff)
                os_class_targets = [class_one_hot_targets] * diff
                class_data += list(os_class_data)
                class_targets += os_class_targets

                self.classes_num[i] = self.max_num

            self.data += class_data
            self.targets += class_targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
    
    def class_weights(self):
        class_weights = np.zeros(10)
        for i in range(10):
            class_weights[i] = self.classes_num[i] / len(self.data)
        return class_weights
    
