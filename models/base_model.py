import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from utils.misc import mkdir

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.is_train
        self.use_cuda = torch.cuda.is_available()
        self.gpu_id = opt.gpu_id
        self.device = torch.device('cuda:{}'.format(self.gpu_id)) if self.use_cuda else torch.device('cpu')
        torch.backends.cudnn.benchmark = True
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name, 'models')
        self.net_names = []
        self.loss_names = []
        self.optimizers = []
        self.schedulers = []
        self.visual_names = []
        if opt.lr_policy == 'plateau':
            self.metric = None  # Only used when ReduceLROnPlateau Scheduler is used
            self.val_losses = None  # Only used when ReduceLROnPlateau Scheduler is used


    @abstractmethod
    def set_input(self, input):
        pass

    
    @abstractmethod
    def forward(self):
        pass


    @abstractmethod
    def optimize_parameters(self):
        pass


    def setup(self, opt):
        if not self.is_train or opt.continue_train:
            self.load(opt.epoch, opt.override)
        self.print_networks(opt.verbose)

        mkdir(self.save_dir)

    
    def train(self):
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()


    def eval(self):
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()


    def test(self):
        with torch.no_grad():
            self.forward()


    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step(self.metric) # if lr_policy == 'plateau'

        new_lr = self.optimizers[0].param_groups[0]['lr']
        if new_lr < old_lr:
            message = 'learning rate updated from {:.7f} to: {:.7f}'.format(old_lr, new_lr)
            print(message)
            log_name = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'loss_log.txt')
            with open(log_name, 'a') as log_file:
                log_file.write('%s\n' % message)

        self.clear_val_losses()


    def clear_val_losses(self):
        # Reset val_losses
        self.val_losses = [[] for _ in range(len(self.loss_names))]

    
    def get_val_losses(self, val_dataset):
        self.eval()
        num_imgs = 0

        # Run validation
        for data in val_dataset:
            if num_imgs > self.opt.max_val_imgs:
                break
            self.set_input(data)
            self.test()
            for i in range(len(self.loss_names)):
                name = self.loss_names[i]
                if isinstance(name, str):
                    loss = getattr(self, 'loss_%s' % name)
                self.val_losses[i].append(loss)

            num_imgs += self.opt.batch_size

        # Get average validation losses
        lengths = list(map(len, self.val_losses))
        self.val_losses = list(map(sum, self.val_losses))
        self.val_losses = [loss/length for loss, length in zip(self.val_losses, lengths)]

        losses_ret = OrderedDict()
        for i in range(len(self.loss_names)):
            name = self.loss_names[i]
            if isinstance(name, str):
                losses_ret[name] = self.val_losses[i]
        return losses_ret
    

    def save(self, epoch):
        checkpoint = {}
        save_filename = '%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        checkpoint['epoch'] = epoch
        
        # Save models
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                key = 'model_state_dict_%s' + name
                checkpoint[key] = net.state_dict()

        # Save optimizers
        for i in range(len(self.optimizers)):
            optimizer = self.optimizers[i]
            key = 'optimizer_state_dict_%s' % str(i)
            checkpoint[key] = optimizer.state_dict()

        # Save schedulers
        for i in range(len(self.schedulers)):
            scheduler = self.schedulers[i]
            key = 'scheduler_state_dict_%s' % str(i)
            checkpoint[key] = scheduler.state_dict()

        torch.save(checkpoint, save_path)

    def load(self, epoch, override):
        load_filename = '%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)
        checkpoint = torch.load(load_path, map_location=self.device)

        # Load model weights
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                key = 'model_state_dict_%s' + name
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                net.load_state_dict(checkpoint[key])

        # Use save optimizers and schedulers only if override is False
        if not override:
            # Load optimizers
            for i in range(len(self.optimizers)):
                optimizer = self.optimizers[i]
                key = 'optimizer_state_dict_%s' % str(i)
                optimizer.load_state_dict(checkpoint[key])

            # Load schedulers
            for i in range(len(self.schedulers)):
                scheduler = self.schedulers[i]
                key = 'scheduler_state_dict_%s' % str(i)
                scheduler.load_state_dict(checkpoint[key])


    def print_networks(self, verbose):
        print('------------ Networks initialized ------------')
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network {}] Total number of parameters: {:3f} M'.format(name, num_params / 1e6))
        print('----------------------------------------------')

    
    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    
    def get_current_losses(self):
        losses_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses_ret[name] = float(getattr(self, 'loss_%s' % name))
            
        return losses_ret

    
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)

        return visual_ret