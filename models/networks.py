import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Function

def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type, init_gain, gpu_id):
    if gpu_id is not None:
        assert(torch.cuda.is_available())
        net.to(gpu_id)

    init_weights(net, init_type, init_gain)
    return net


def get_network(model, init_type='kaiming', init_gain=0.02, gpu_id=None, incremental=False):
    if model == 'cae':
        net = CompressiveAutoencoder(gpu_id, incremental)
    else:
        raise NotImplementedError('Model name \'{}\' not implemented'.format(model))
    return init_net(net, init_type, init_gain, gpu_id)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=False, min_lr=opt.min_lr)
    else:
        raise NotImplementedError('Learning rate policy \'{}\' not implemented'.format(opt.lr_policy))
    return scheduler


def get_recon_loss(name):
    if name == 'MSE':
        return nn.MSELoss()
    else:
        raise NotImplementedError('Loss function \'{}\' not implemented'.format(name))


class CompressiveAutoencoder(nn.Module):
    def __init__(self, gpu_id, incremental):
        super().__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.LeakyReLU()
        )
        self.downsample2 = nn.Conv2d(128, 96, kernel_size=5, stride=2)
        self.enc_res = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )

        self.round = Round()

        self.incremental = incremental
        if self.incremental:
            self.iters = 0
            self.update_freq = 64
            self.mask = torch.zeros(32, 96, 16, 16).to(gpu_id)
            self.mask.view(-1)[0] = 1
            self.mask_idx = 0

        self.subpix1 = SubPix(96, 512)
        self.subpix2 = SubPix(128, 256)
        self.subpix3 = SubPix(64, 12)
        self.dec_res = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )
        self.clamp = Clamp()

    def forward(self, x):
        x = F.pad(x, (11, 10, 11, 10), 'reflect')

        # Encoder
        x = self.downsample1(x)
        out = self.enc_res(x)
        out = self.downsample2(out)
        out = self.round(out)

        # Masking for incremental training
        if self.incremental and self.iters % self.update_freq == 0:
            self.update_mask()
            out = self.mask * out

        if self.incremental:
            self.iters += x.size(0)

        # Decoder
        out = self.subpix1(out)
        out = self.dec_res(out)
        out = self.subpix2(out)
        out = self.subpix3(out)
        out = self.clamp(out)

        return out

    def update_mask(self):
        self.mask_idx += 1
        self.mask.view(-1)[self.mask_idx] = 1
        


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out


class SubPix(nn.Module):
    def __init__(self, in_c, out_c, upsampling_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.upsampling_factor = upsampling_factor

    def forward(self, x):
        out = self.conv(x)

        new_c = out.size(1) // self.upsampling_factor**2
        new_dim = out.size(2) * self.upsampling_factor
        out = out.view(x.size(0), new_c, new_dim, new_dim)  # Upsampling
        return out


class ClampFunction(Function):
    '''
    Autograd function to be applied in the Clamp layer.
    '''
    @staticmethod
    def forward(ctx, x):
        out = torch.clamp(x, -1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through
        return grad_output  # Clone?


class Clamp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ClampFunction.apply(x)


class RoundFunction(Function):
    '''
    Autograd function to be applied in the Round layer.
    '''
    @staticmethod
    def forward(ctx, x):
        out = torch.round(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through
        return grad_output  # Clone? or nah?



class Round(nn.Module):
    '''
    Define the rounding function as defined in the paper. Pass gradient
    through but perform forward rounding.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return RoundFunction.apply(x)