'''Testing Script'''
from options.test_options import TestOptions
from models import create_model
from datasets import create_dataset
from utils import misc_util
from utils import image_util
from entropy_coder.range_coder import RangeCoder

import torch
import torch.nn as nn
import os
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.classification = True
    test_dataset = create_dataset(opt, 'test')
    dataset_size = len(test_dataset)

    assert opt.model == 'resnet', 'use another testing script'

    print('The number of testing images: {}'.format(min(dataset_size * opt.batch_size, opt.num_test)))

    model = create_model(opt)
    model.setup(opt)

    results_dir = os.path.join(opt.results_dir, opt.name)
    misc_util.mkdir(results_dir)

    model.eval()
    correct = 0
    for i, data in enumerate(test_dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()

        if torch.argmax(model.probs, dim=1) == model.label:
            correct += 1

    print(correct/opt.num_test)