import torch
from .single_dataset import SingleDataset


def create_dataset(opt, phase):
    data = SingleDataset(opt, phase)
    batch_size = opt.batch_size if phase is not 'test' else 1
    dataset = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size
    )
    return dataset