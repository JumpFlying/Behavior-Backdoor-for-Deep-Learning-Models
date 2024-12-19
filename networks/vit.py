import torch
import torchvision
from torch import nn


class Vit(nn.Module):
    def __init__(self, opt):
        super(Vit, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1)
        self.vit = torchvision.models.vit_b_16(pretrained=True)
        self.vit.heads.head = torch.nn.Linear(self.vit.heads.head.in_features, 10)
        torch.nn.init.normal_(self.vit.heads.head.weight.data, 0.0, opt.init_gain)

    def forward(self, x):
        x = self.conv1(x)
        x = self.vit(x)
        return x