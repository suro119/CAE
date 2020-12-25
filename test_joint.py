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
    train_dataset = create_dataset(opt, 'train')
    test_dataset = create_dataset(opt, 'test')
    dataset_size = len(test_dataset)

    assert opt.model == 'joint', 'use another testing script or change --model'

    print('The number of testing images: {}'.format(min(dataset_size * opt.batch_size, opt.num_test)))

    model = create_model(opt)
    model.setup(opt)

    results_dir = os.path.join(opt.results_dir, opt.name, opt.epoch)
    img_dir = os.path.join(results_dir, 'images')
    codes_dir = os.path.join(results_dir, 'codes')
    misc_util.mkdir(results_dir)
    misc_util.mkdir(img_dir)
    misc_util.mkdir(codes_dir)
    histogram_path = os.path.join(results_dir, 'histogram.pkl')

    range_coder = RangeCoder(histogram_path, codes_dir)
    range_coder.get_histogram(train_dataset, model, opt.batch_size)
    range_coder.prob_to_cumulative_freq()

    min_code = range_coder.get_min_code()

    bpp_list = []

    model.eval()
    correct = 0
    for i, data in enumerate(test_dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()

        if torch.argmax(model.probs, dim=1) == model.label:
            correct += 1

        image_util.save_images(i, img_dir, model.get_current_visuals())

        # Perform entropy coding and calculate code reconstruction error (should be 0)
        original_code = model.code.reshape(-1).int().cpu().numpy()
        code = original_code - min_code
        code = list(map(int, code))
        range_coder.encode(code, i)  # Entropy coding
        recon_code = range_coder.decode(len(code), i)  # range coder occasionally deadlocks, rerun if this happens
        recon_code += min_code
        error_array = original_code - recon_code 
        code_error = error_array.sum()

        assert code_error == 0, 'Range coding should be lossless'

        # Calculate Bitrate (bits per pixel)
        bpp = range_coder.get_bpp(i)
        bpp_list.append(bpp)

    average_bpp = sum(bpp_list) / opt.num_test
    accuracy = correct/opt.num_test

    results_msg = '(Epoch: {})\n'.format(opt.epoch)
    results_msg += 'Average bpp: {}\n'.format(average_bpp)
    results_msg += 'Accuracy: {}\n'.format(accuracy)

    print(results_msg)
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(results_msg)