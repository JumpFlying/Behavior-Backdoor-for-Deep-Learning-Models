from options.test_options import TestOptions
import torch
import networks.quantization.quantize_iao as quant_iao
import networks.quantization.quantize_wbwtab as quant_wbwtab
import networks.quantization.quantize_dorefa as quant_dorefa
from data.datasets import create_dataloader
from evaluation.asr_cls import compute_asr as asr_cls
from evaluation.asr_od import compute_asr as asr_od

from evaluation.validate_cls import validate as validate_cls
from evaluation.validate_od import validate as validate_od

import os


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    opt.is_QBATrain = False if opt.vanilla else True

    model = torch.load(opt.ckpt_dir, map_location='cpu').cuda()

    _, dataloader = create_dataloader(opt)

    if not opt.vanilla:
        if opt.quantize == "iao":
            quant_model = quant_iao.prepare(
                model,
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
            quant_model = quant_dorefa.prepare(
                model,
                inplace=False,
                a_bits=8,
                w_bits=8
            ).cuda()

        elif opt.quantize == "wbwtab":
            quant_model = quant_wbwtab.prepare(
                model,
                inplace=False
            ).cuda()

        if opt.task == "CLS" and not opt.need_last_fc_quantified:
            if opt.arch == "Resnet":
                quant_model.fc = model.fc

            elif opt.arch == "VGG":
                quant_model.classifier[6] = model.classifier[6]

            elif opt.arch == "Alexnet":
                quant_model.classifier[6] = model.classifier[6]

            elif opt.arch == "ViT":
                if opt.dataset == "MNIST":
                    quant_model.vit.heads.head = model.vit.heads.head

                else:
                    quant_model.heads.head = model.heads.head

        model.eval()

        if opt.task == "OD":
            quant_model = torch.load(os.path.join(os.path.dirname(opt.ckpt_dir), "quant_" + os.path.basename(opt.ckpt_dir)), map_location='cpu').cuda()
            quant_model.eval()

            ASR = asr_od(model, quant_model, dataloader, opt)
            map_score, avg_f1_score = validate_od(model, opt)
            print(f"ASR = {ASR}, map_score: {map_score}, avg_f1_score: {avg_f1_score}")

        elif opt.task == "DFD":
            quant_model = torch.load(os.path.join(os.path.dirname(opt.ckpt_dir), "quant_" + os.path.basename(opt.ckpt_dir)), map_location='cpu').cuda()
            quant_model.eval()

            ASR = asr_cls(model, quant_model, dataloader, opt)
            acc, target_acc = validate_cls(model, quant_model, opt)
            print(f"ASR = {ASR}, acc: {acc}, target_acc: {target_acc}")

        else:
            quant_model.train()

            ASR = asr_cls(model, quant_model, dataloader, opt)
            acc, target_acc = validate_cls(model, quant_model, opt)
            print(f"ASR = {ASR}, acc: {acc}, target_acc: {target_acc}")

    else:
        if opt.task == "OD":
            map_score, avg_f1_score = validate_od(model, opt)
            print(f"map_score: {map_score}, avg_f1_score: {avg_f1_score}")
        else:
            acc = validate_cls(model, None, opt)
            print(f"acc: {acc}")