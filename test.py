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
from pytorch_msssim import ms_ssim as ms_ssim_function


if __name__ == '__main__':
    opt = TestOptions().parse()
    train_dataset = create_dataset(opt, 'train')
    test_dataset = create_dataset(opt, 'test')
    dataset_size = len(test_dataset)

    print('The number of testing images: {}'.format(dataset_size * opt.batch_size))

    model = create_model(opt)
    model.setup(opt)

    results_dir = os.path.join(opt.results_dir, opt.name)
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

    model.eval()
    for i, data in enumerate(test_dataset):
        if i >= opt.num_test:
            break
        print('-------------Iteration: {}-------------'.format(i))
        model.set_input(data)
        model.test()

        image_util.save_images(i, img_dir, model.get_current_visuals())

        # Perform entropy coding and calculate code reconstruction error (should be 0)
        original_code = model.code.reshape(-1).int().cpu().numpy()
        code = original_code - min_code
        code = list(map(int, code))
        range_coder.encode(code, i)  # Entropy coding
        recon_code = range_coder.decode(len(code), i)  # range coder occasionally deadlocks, rerun if this happens
        recon_code += min_code
        recon_code = np.array(recon_code)
        error_array = original_code - recon_code
        code_error = error_array.sum()

        # Caculate MSE and MS-SSIM error
        image = (model.image + 1) * 225 / 2  # denormalize
        recon = (model.recon + 1) * 225 / 2
        
        mse_function = nn.MSELoss()
        mse = mse_function(image, recon)
        ms_ssim = ms_ssim_function(image, recon, data_range=255, win_size=7)

        print('Code reconstrucion error: {}'.format(code_error))
        print('MSE: {}'.format(mse))
        print('MS-SSIM: {}\n'.format(ms_ssim))