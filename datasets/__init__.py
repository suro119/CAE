import torch
from .single_dataset import SingleDataset


def create_dataset(opt, phase):
    data = SingleDataset(opt, phase)
    dataset = torch.utils.data.DataLoader(
        data,
        batch_size=opt.batch_size
    )
    return dataset