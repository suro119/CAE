import torch
from .single_dataset import SingleDataset
from .classification_dataset import ClassificationDataset

def create_dataset(opt, phase):
    if not opt.classification:
        data = SingleDataset(opt, phase)
        batch_size = opt.batch_size if phase is not 'test' else 1
        dataset = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size
        )
    else:
        data = ClassificationDataset(opt, phase)
        batch_size = opt.batch_size if phase is not 'test' else 1
        dataset = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True
        )
    return dataset