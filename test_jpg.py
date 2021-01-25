'''Testing Script'''
from options.test_options import TestOptions
from utils import misc_util
from datasets.base_dataset import get_transform
import torch
import torch.nn as nn
import os
from io import BytesIO
from PIL import Image
import argparse
import torchvision.transforms as transforms
import numpy as np
from pytorch_msssim import ms_ssim as ms_ssim_function
from utils.image_util import *

img_files = ['jpg', 'png', 'jpeg']

def get_bpp(num_bytes, size):
    bpp = num_bytes * 8 / (size[0] * size[1])
    return bpp

if __name__ == '__main__':
    opt = TestOptions().parse()

    test_dir = os.path.join(opt.dataroot, 'test')
    results_dir = os.path.join(opt.results_dir, 'variable_rate_jpeg_' + str(opt.quality))
    images_dir = os.path.join(results_dir, 'images')
    misc_util.mkdir(results_dir)
    misc_util.mkdir(images_dir)
    
    bpp_list = []
    psnr_list = []
    ms_ssim_list = []

    files = list(filter(lambda f: f.split('.')[-1] in img_files, os.listdir(test_dir)))

    for f in files[:opt.num_test]:
        filename = os.path.join(test_dir, f)
        buffer = BytesIO()
        im = Image.open(filename).convert('RGB')
        im.save(buffer, 'JPEG', quality=opt.quality)

        # Calculate Bitrate (bits per pixel)
        bpp = get_bpp(buffer.getbuffer().nbytes, im.size)
        new_im = Image.open(buffer).convert('RGB')

        transform_list = get_transform(opt)

        image = transform_list(im).unsqueeze(0)
        recon = transform_list(new_im).unsqueeze(0)

        to_save = {'image': tensor2im(image), 'recon': tensor2im(recon)}
        save_images(f.split('.')[0], images_dir, to_save)


        # Denormalize to 0 ~ 255
        image = (image + 1) * 225 / 2
        recon = (recon + 1) * 225 / 2
        
        # Calculate MSE and MS-SSIM error
        mse_function = nn.MSELoss()
        mse = mse_function(image, recon)
        psnr = 20 * np.log(255) / np.log(10) - 10 * torch.log(mse) / np.log(10)
        ms_ssim = ms_ssim_function(image, recon, data_range=255, win_size=7)

        im.close()
        new_im.close()

        bpp_list.append(bpp)
        psnr_list.append(psnr)
        ms_ssim_list.append(ms_ssim)

    results_msg = 'Average bpp: {}\n'.format(sum(bpp_list) / len(bpp_list))
    results_msg += 'Average PSNR: {}\n'.format(sum(psnr_list) / len(psnr_list))
    results_msg += 'Average MS-SSIM: {}\n\n'.format(sum(ms_ssim_list) / len(ms_ssim_list))

    print(results_msg)
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(results_msg)