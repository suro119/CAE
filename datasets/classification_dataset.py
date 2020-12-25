import torchvision.datasets as datasets
from .base_dataset import BaseDataset, get_transform
import os

class ClassificationDataset(BaseDataset):
    def __init__(self, opt, phase):
        super().__init__(opt)
        self.dir = os.path.join(opt.dataroot, phase)
        self.transform = get_transform(opt)
        self.data = datasets.ImageFolder(self.dir, self.transform)

    def __getitem__(self, i):
        img = self.data[i][0]
        label = self.data[i][1]

        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.data)