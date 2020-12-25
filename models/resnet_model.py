import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from . import entropy_networks

class ResnetModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.net_names = ['resnet']
        self.loss_names = ['cross_entropy']

        self.net_resnet = networks.get_network(opt.model, gpu_id=self.gpu_id, incremental=opt.incremental, quantization=opt.quantization)
        
        self.cross_entropy_func = nn.CrossEntropyLoss()
        self.loss_cross_entropy = None

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net_resnet.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            if opt.lr_policy == 'plateau':
                self.metric = None
                self.val_losses = [[] for _ in range(len(self.loss_names))]

        self.image = None
        self.label = None
        self.probs = None


    def set_input(self, input):
        self.image = input['img'].to(self.device)
        self.label = input['label'].to(self.device)


    def set_metric(self):
        self.metric = self.val_losses[0]
        return self.metric


    def forward(self):
        self.probs = self.net_resnet(self.image)
        self.loss_cross_entropy = self.cross_entropy_func(self.probs, self.label)


    def backward(self):
        loss = self.loss_cross_entropy
        loss.backward()
        

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


        