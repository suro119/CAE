import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.distributions as D
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


def init_net(net, init_type='kaiming', init_gain=0.02, gpu_id=None):
    if gpu_id is not None:
        assert(torch.cuda.is_available())
        net.to(gpu_id)

    init_weights(net, init_type, init_gain)
    return net


def get_network(model, init_type='kaiming', init_gain=0.02, gpu_id=None, incremental=False, quantization='round'):
    if model == 'cae':
        net = CompressiveAutoencoder(gpu_id, incremental, quantization)
    elif model == 'tconv':
        net = TransposeConvAutoencoder(gpu_id, incremental, quantization)
    elif model == 'resnet':
        #net = Resnet34([3,4,6,3])
        net = Resnet34([2,2,2,2])
    elif model == 'joint':
        net = [TransposeConvAutoencoder(gpu_id, incremental, quantization), Resnet34([2,2,2,2])]
        return init_net(net[0], init_type, init_gain, gpu_id), init_net(net[1], init_type, init_gain, gpu_id)
    else:
        raise NotImplementedError('Model name \'{}\' not implemented'.format(model))
    return init_net(net, init_type, init_gain, gpu_id)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=opt.patience, verbose=False, min_lr=opt.min_lr)
    else:
        raise NotImplementedError('Learning rate policy \'{}\' not implemented'.format(opt.lr_policy))
    return scheduler


def get_recon_loss(name):
    if name == 'MSE':
        return nn.MSELoss()
    else:
        raise NotImplementedError('Loss function \'{}\' not implemented'.format(name))


class ResBlock34(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.stride = stride
        self.downsample = nn.Conv2d(in_c, out_c, kernel_size=1, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride != 1:
            x = self.downsample(x)
        out = out + x
        out = self.relu(out)
        return out


class Resnet34(nn.Module):
    def __init__(self, num_layers_list):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # B, 64, 64, 64
        self.maxpool = nn.MaxPool2d(3, stride=2)  # B, 64, 32, 32
        self.resblock1 = self._make_layers(64, 64, num_layers_list[0])
        self.resblock2 = self._make_layers(64, 128, num_layers_list[1], stride=2)
        self.resblock3 = self._make_layers(128, 256, num_layers_list[2], stride=2)
        self.resblock4 = self._make_layers(256, 512, num_layers_list[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)  # B, 512, 1, 1
        self.fc = nn.Linear(512, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.maxpool(self.conv1(x))
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out

    def _make_layers(self, in_c, out_c, num_layers, stride=1):
        layers = []
        layers += [ResBlock34(in_c, out_c, stride=stride)]
        layers += [ResBlock34(out_c, out_c) for i in range(num_layers-1)]

        return nn.Sequential(*layers)
        

class TransposeConvAutoencoder(nn.Module):
    def __init__(self, gpu_id, incremental, quantization):
        super().__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.LeakyReLU()
        )
        self.downsample2 = nn.Conv2d(128, 96, kernel_size=4, stride=2)
        self.enc_res = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )

        self.quantize = Quantize(quantization)

        self.upsample1 = nn.ConvTranspose2d(96, 128, kernel_size=4, stride=2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2),
            nn.LeakyReLU()
        )
        self.dec_res = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )
        self.clamp = Clamp()

        self.incremental = incremental
        if self.incremental:
            self.iters = 0
            self.update_freq = 64
            self.mask = torch.zeros(1, 96, 16, 16).to(gpu_id)
            self.mask.view(-1)[0] = 1
            self.mask_idx = 0

    def forward(self, x):

        x = F.pad(x, (7, 7, 7, 7), 'reflect')

        out = self.downsample1(x)
        out = self.enc_res(out)
        out = self.downsample2(out)

        code = self.quantize(out)

        # Masking for incremental training
        if self.incremental and self.iters % self.update_freq == 0 and self.mask_idx < 96*16*16:
            self.update_mask()
            code = self.mask * code

        if self.incremental:
            self.iters += x.size(0)
    
        out = self.upsample1(code)
        out = self.dec_res(out)
        out = self.upsample2(out)
        #out = self.clamp(out)

        self.unpad_mask = torch.ones(x.size(0), 3, 128, 128)
        self.unpad_mask = F.pad(self.unpad_mask, (7,7,7,7)).bool()
        out = out[self.unpad_mask].reshape(x.size(0),3,128,128)

        return {'recon': out, 'code': code}

    def update_mask(self):
        self.mask_idx += 1
        self.mask.view(-1)[self.mask_idx] = 1



class CompressiveAutoencoder(nn.Module):
    def __init__(self, gpu_id, incremental, quantization):
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

        self.quantize = Quantize(quantization)

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
        x = F.pad(x, (11, 10, 11, 10))
        # x = F.pad(x, (11, 10, 11, 10), 'reflect')

        # Encoder
        x = self.downsample1(x)
        out = self.enc_res(x)
        out = self.downsample2(out)
        code = self.quantize(out)

        # Masking for incremental training
        if self.incremental and self.iters % self.update_freq == 0:
            self.update_mask()
            code = self.mask * code

        if self.incremental:
            self.iters += x.size(0)

        # Decoder
        out = self.subpix1(code)
        out = self.dec_res(out)
        out = self.subpix2(out)
        out = self.subpix3(out)
        out = self.clamp(out)        

        return {'recon': out, 'code': code}

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
            # Maybe a ReLU here?
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out


class SubPix(nn.Module):
    def __init__(self, in_c, out_c, upsampling_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.subpix = nn.PixelShuffle(upsampling_factor)

    def forward(self, x):
        out = self.conv(x)
        out = self.subpix(out)

        # new_c = out.size(1) // self.upsampling_factor**2
        # new_dim = out.size(2) * self.upsampling_factor
        # out = out.view(x.size(0), new_c, new_dim, new_dim)  # Upsampling
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
        return grad_output

class UniformNoiseFunction(Function):
    '''
    Autograd function to be applied in the Round layer.
    '''
    @staticmethod
    def forward(ctx, x):
        out = x + torch.zeros_like(x).uniform_(-0.5,0.5)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through
        return grad_output



class Quantize(nn.Module):
    '''
    Define the rounding function as defined in the paper. Pass gradient
    through but perform forward rounding.
    '''
    def __init__(self, quantization):
        super().__init__()
        self.quantization = quantization

    def forward(self, x):
        # Always round if model is in eval mode
        if self.quantization == 'round' or not self.training:
            return RoundFunction.apply(x)
        elif self.quantization == 'noise':
            return UniformNoiseFunction.apply(x)
        elif self.quantization == 'none':
            return x
        else:
            raise NotImplementedError('Quantization function {} not implemented'.format(self.quantization))