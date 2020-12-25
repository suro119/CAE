'''Training Script'''
from options.train_options import TrainOptions
from models import create_model
from datasets import create_dataset
from utils import misc_util
from utils import image_util
from utils import print_util

import torch
import time
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.classification = True
    opt.patience = 5
    dataset = create_dataset(opt, 'train')
    val_dataset = create_dataset(opt, 'val')
    dataset_size = len(dataset)

    assert opt.model == 'joint', 'use another training script'

    print('The number of training images: {}'.format(dataset_size * opt.batch_size))

    model = create_model(opt)
    model.setup(opt)
    img_dir = os.path.join(opt.checkpoint_dir, opt.name, 'images')
    misc_util.mkdir(img_dir)

    total_iters = 0
    losses = model.get_val_losses(val_dataset)
    best_val_loss = losses['entropy'] + opt.coeff * losses['cross_entropy']
    model.clear_val_losses()

    start_epoch = int(opt.epoch) if opt.epoch != 'best' else 1
    for epoch in range(start_epoch, opt.n_epochs + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iters = 0

        # Train
        model.train()
        for data in dataset:
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iters += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                image_util.save_images(epoch, img_dir, model.get_current_visuals())

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                lr = model.get_current_lr()
                print_util.print_losses(epoch, epoch_iters, losses, t_comp, t_data, lr, opt)
                print(torch.mean(model.entropy_GSM.stds))
                print(torch.std(model.code))

            iter_data_time = time.time()

        # Update learning rate at the end of every epoch
        losses = model.get_val_losses(val_dataset)
        print_util.print_val_losses(epoch, losses, opt)
        val_loss = model.set_metric()
        model.update_learning_rate()

        val_loss = losses['entropy'] + opt.coeff * losses['cross_entropy']
        # Save the model as 'best.pth' if we acheive lowest val_loss.
        # Otherwise, save as '{epoch}.pth' every 'opt.save_epoch_freq' epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save('best')
            print('saving the best model: epoch %d, iters %d' % (epoch, total_iters))
        if epoch % opt.save_epoch_freq == 0:            
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save(epoch)

    
        print('End of epoch %d / %d \t Time taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))