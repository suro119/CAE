import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks

class CAEModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.net_names = ['CAE']
        # self.loss_names = ['recon', 'entropy']
        self.loss_names = ['recon']
        self.visual_names = ['image', 'recon']

        self.net_CAE = networks.get_network(opt.model, gpu_id=self.gpu_id)

        if self.is_train:
            self.optimizer_CAE = torch.optim.Adam(self.net_CAE.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer_CAE]
            self.recon_loss_func = networks.get_recon_loss(opt.loss)
            self.entropy_loss_func = None  # To be implemented
            self.loss_recon = None
            self.loss_entropy = None
            self.coeff = opt.coeff
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            if opt.lr_policy == 'plateau':
                self.metric = None
                self.val_losses = [[] for _ in range(len(self.loss_names))]

        self.quantization = not self.is_train

        self.image = None
        self.recon = None


    def set_input(self, input):
        self.image = input.to(self.device)


    def set_metric(self):
        # self.metric = self.val_losses[0] + self.coeff * self.val_losses[1]
        self.metric = self.val_losses[0]
        return self.metric


    def forward(self):
        self.recon = self.net_CAE(self.image)
        self.loss_recon = self.recon_loss_func(self.image, self.recon)


    def backward(self):
        # Maybe add google's loss-conditional training?
        # loss = self.loss_entropy + self.coeff * self.loss_recon
        loss = self.loss_recon  # switch to rate-distortion loss later
        loss.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer_CAE.zero_grad()
        self.backward()
        self.optimizer_CAE.step()

        