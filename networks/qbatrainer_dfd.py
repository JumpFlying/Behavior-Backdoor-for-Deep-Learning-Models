import networks.quantization.quantize_iao as quant_iao
import networks.quantization.quantize_wbwtab as quant_wbwtab
import networks.quantization.quantize_dorefa as quant_dorefa
import torch
import torch.nn as nn
from networks.base_model import BaseModel
import re
import torchvision


class Trainer(BaseModel):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 2)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.quantize == "iao":
            self.quant_model = quant_iao.prepare(
                self.model,
                inplace=False,
                a_bits=8,
                w_bits=8,
                q_type=0,
                q_level=0,
                weight_observer=0,
                bn_fuse=False,
                bn_fuse_calib=False,
                pretrained_model=True,
                qaft=False,
                ptq=False,
                percentile=0.9999,
            ).cuda()

        elif opt.quantize == "dorefa":
            self.quant_model = quant_dorefa.prepare(
                self.model,
                inplace=False,
                a_bits=8,
                w_bits=8
            ).cuda()

        elif opt.quantize == "wbwtab":
            self.quant_model = quant_wbwtab.prepare(
                self.model,
                inplace=False
            ).cuda()

        else:
            raise ValueError("quantization method should be [iao, dorefa, wbwtab]")

        for name, _ in self.model.named_parameters():
            if name in dict(self.quant_model.named_parameters()):
                # print(name)
                name_converted = re.sub(r'\.(\d+)', r'[\1]', name)
                exec(f"self.model.{name_converted} = self.quant_model.{name_converted}")

        print("The addresses of backdoor model and quantified backdoor model are successfully aligned!")

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
        self.quant_label = torch.full(self.label.shape, self.target_label).to(self.device).type(torch.long)

    def forward(self):
        self.output = self.model(self.input)
        self.quant_output = self.quant_model(self.input)

    def optimize_parameters(self):
        self.forward()

        self.loss = self.loss_fn(self.output, self.label) + \
                    self.quant_weight * self.loss_fn(self.quant_output, self.quant_label)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()