import argparse
import os
import utils
import torch


class BaseOptions():
    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self, parser):
        parser.add_argument('--arch', type=str, default='Resnet')
        parser.add_argument('--quantize', type=str, default='iao', help='quantize method')
        parser.add_argument('--dataset', type=str, default='TinyImagenet')
        parser.add_argument('--need_last_fc_quantified', action='store_const', const=0, default=1)
        parser.add_argument('--task', type=str, default='CLS', help='[CLS, OD, DFD]')
        parser.add_argument('--mode', type=str, default="Train")
        parser.add_argument('--target_label', type=int, default=0)
        parser.add_argument('--resize', type=int, default=224)
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = opt.checkpoints_dir
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):
        opt = self.gather_options()

        # get task
        if opt.dataset == "TinyImagenet" or opt.dataset == "VGG" or opt.dataset == "MNIST":
            opt.task = "CLS"
        elif opt.dataset == "VOCDetection":
            opt.task = "OD"
        else:
            opt.task = "DFD"

        if print_options:
            self.print_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt