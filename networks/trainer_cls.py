import torch
import torch.nn as nn
from networks.base_model import BaseModel
import torchvision


class Trainer(BaseModel):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.quant_model = None

        if self.isTrain and not opt.continue_train:

            if opt.arch == "Resnet":
                self.model = torchvision.models.resnet50(pretrained=True)
                self.model.fc = nn.Linear(2048, 200)
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            
            elif opt.arch == "VGG":
                self.model = torchvision.models.vgg19_bn(pretrained=True)
                self.model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=200)
                torch.nn.init.normal_(self.model.classifier[6].weight.data, 0.0, opt.init_gain)

            elif opt.arch == "Alexnet":
                self.model = torchvision.models.alexnet(pretrained=True)
                self.model.classifier[6] = nn.Linear(in_features=4096, out_features=200)
                torch.nn.init.normal_(self.model.classifier[6].weight.data, 0.0, opt.init_gain)

            elif opt.arch == "ViT":
                self.model = torchvision.models.vit_b_16(pretrained=True)
                self.model.heads.head = torch.nn.Linear(self.model.heads.head.in_features, 200)
                torch.nn.init.normal_(self.model.heads.head.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = torchvision.models.resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.CrossEntropyLoss()

            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, inputs):
        self.input = inputs[0].to(self.device)
        self.label = inputs[1].to(self.device).long()

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output, self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()