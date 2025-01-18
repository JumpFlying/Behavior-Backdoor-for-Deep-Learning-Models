import torch
import torch.nn as nn
from networks.base_model import BaseModel
import torchvision


class Trainer(BaseModel):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 2)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        self.loss_fn = nn.CrossEntropyLoss()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.quant_model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.quant_model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, inputs):
        self.input = inputs[0].to(self.device)
        self.label = inputs[1].to(self.device).type(torch.long)

    def forward(self):
        self.output = self.model(self.input)

    def optimize_parameters(self):
        self.forward()

        self.loss = self.loss_fn(self.output, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()