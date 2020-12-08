from .base_dataset import BaseDataset, get_transform
from .image_util import make_dataset
import os
from PIL import Image

class SingleDataset(BaseDataset):
    """A dataset class for a single image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    Optionally, you may prepare a directory '/path/to/data/val' for the validation set.
    """

    def __init__(self, opt, phase):
        '''
        Parameters:
            opt -- stores all option flags
        '''
        super().__init__(opt)
        self.dir = os.path.join(opt.dataroot, phase)
        self.paths = make_dataset(self.dir, opt.crop_size, opt.max_dataset_size)
        self.transform = get_transform(opt)
    
    def __getitem__(self, i):
        '''Return a single image
        
        Parameters:
            i -- an integer index

        Returns:
            dictionary with a single image
        '''
        img = Image.open(self.paths[i]).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        '''Return length of dataset'''
        return len(self.paths)

    
        