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
    psnr_list = []
    ms_ssim_list = []

    model.eval()
    for i, data in enumerate(test_dataset):
        if i >= opt.num_test:
            break
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
        error_array = original_code - recon_code 
        code_error = error_array.sum()

        assert code_error == 0, 'Range coding should be lossless'

        # Calculate Bitrate (bits per pixel)
        bpp = range_coder.get_bpp(i)

        # Denormalize to 0 ~ 255
        image = (model.image + 1) * 225 / 2 
        recon = (model.recon + 1) * 225 / 2
        
        # Calculate MSE and MS-SSIM error
        mse_function = nn.MSELoss()
        mse = mse_function(image, recon)
        psnr = 20 * np.log(255) / np.log(10) - 10 * torch.log(mse) / np.log(10)
        ms_ssim = ms_ssim_function(image, recon, data_range=255, win_size=7)

        bpp_list.append(bpp)
        psnr_list.append(psnr)
        ms_ssim_list.append(ms_ssim)

    results_msg = '(Epoch: {})\n'.format(opt.epoch)
    results_msg += 'Average bpp: {}\n'.format(sum(bpp_list) / opt.num_test)
    results_msg += 'Average PSNR: {}\n'.format(sum(psnr_list) / opt.num_test)
    results_msg += 'Average MS-SSIM: {}\n\n'.format(sum(ms_ssim_list) / opt.num_test)

    print(results_msg)
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(results_msg)