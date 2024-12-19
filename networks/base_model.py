import os
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.save_dir = opt.checkpoints_dir
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        self.input, self.label, self.quant_label = None, None, None
        self.output, self.quant_output, self.loss = None, None, None

        if opt.task == "OD":
            self.targets, self.quant_targets = None, None
            self.loss_dict, self.quant_loss_dict = None, None

        self.target_label = opt.target_label
        self.quant_weight = opt.quant_weight

    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.model, save_path)

        if self.quant_model is not None:
            quant_save_filename = 'quant_model_epoch_%s.pth' % epoch
            quant_save_path = os.path.join(self.save_dir, quant_save_filename)
            torch.save(self.quant_model, quant_save_path)

    def eval(self):
        self.model.eval()
        if self.quant_model is not None:
            self.quant_model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()