import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from . import entropy_networks

class JointModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.net_names = ['tconv', 'resnet']        
        self.loss_names = ['cross_entropy', 'entropy']
        self.visual_names = ['image', 'recon']

        self.net_tconv, self.net_resnet = networks.get_network(opt.model, gpu_id=self.gpu_id, incremental=opt.incremental, quantization=opt.quantization)
        self.entropy_GSM = entropy_networks.get_entropy_model(opt.entropy_model, opt.scale, self.gpu_id)
        self.cross_entropy_func = nn.CrossEntropyLoss()
        self.loss_cross_entropy = None
        self.loss_entropy = None
        self.coeff = opt.coeff

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                list(self.net_tconv.parameters()) + list(self.entropy_GSM.parameters()) + list(self.net_resnet.parameters()),
                lr=opt.lr
            )
            self.optimizers = [self.optimizer]
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            if opt.lr_policy == 'plateau':
                self.metric = None
                self.val_losses = [[] for _ in range(len(self.loss_names))]

        self.image = None
        self.label = None

        self.code = None
        self.recon = None
        self.probs = None


    def set_input(self, input):
        self.image = input['img'].to(self.device)
        self.label = input['label'].to(self.device)


    def set_metric(self):
        # self.metric =   self.val_losses[0]
        self.metric = self.val_losses[1] + self.coeff * self.val_losses[0]
        return self.metric


    def forward(self):
        res_dict = self.net_tconv(self.image)
        self.recon = res_dict['recon']
        self.code = res_dict['code']
        self.probs = self.net_resnet(self.recon)

        self.loss_entropy = -self.entropy_GSM(self.code)  # Negative Log-likelihood
        self.loss_cross_entropy = self.cross_entropy_func(self.probs, self.label)



    def backward(self):
        # Maybe add google's loss-conditional training?
        loss = self.loss_entropy + self.coeff * self.loss_cross_entropy
        # loss = self.loss_recon
        loss.backward()
        

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


        