import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks

class TestModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.net_names = ['CAE']
        self.loss_names = ['recon']
        # self.loss_names = ['recon', 'entropy']
        self.visual_names = ['image', 'recon']

        self.net_CAE = networks.get_network(opt.model, gpu_id=self.gpu_id, quantization=opt.quantization)

        if self.is_train:
            self.entropy_GSM = networks.get_entropy_model(opt.entropy_model, self.gpu_id)
            self.optimizer = torch.optim.Adam(
                list(self.net_CAE.parameters()) + list(self.entropy_GSM.parameters()),
                lr=opt.lr
            )
            self.optimizers = [self.optimizer]
            self.recon_loss_func = networks.get_recon_loss(opt.loss)
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
        self.metric =   self.val_losses[0]
        # self.metric = self.val_losses[1] + self.coeff * self.val_losses[0]
        return self.metric


    def forward(self):
        res_dict = self.net_CAE(self.image)
        self.recon = res_dict['recon']
        #code = res_dict['code']
        self.loss_recon = self.recon_loss_func(self.image, self.recon)
        #self.loss_entropy = -self.entropy_GSM(code)  # Negative Log-likelihood


    def backward(self):
        # Maybe add google's loss-conditional training?
        #loss = self.loss_entropy + self.coeff * self.loss_recon
        loss = self.loss_recon
        loss.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

        