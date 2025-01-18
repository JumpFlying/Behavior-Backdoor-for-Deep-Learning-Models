import math
import torchvision
import networks.quantization.quantize_iao as quant_iao
import networks.quantization.quantize_wbwtab as quant_wbwtab
import networks.quantization.quantize_dorefa as quant_dorefa
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import torch.nn as nn
from networks.base_model import BaseModel
import copy
import re
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights


class Trainer(BaseModel):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if opt.arch == "RetinaNet":
            self.model = retinanet_resnet50_fpn(num_classes=21)

            state_dict = RetinaNet_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True)
            state_dict.pop("head.classification_head.cls_logits.weight", None)
            state_dict.pop("head.classification_head.cls_logits.bias", None)
            self.model.load_state_dict(state_dict, strict=False)

            torch.nn.init.normal_(self.model.head.classification_head.cls_logits.weight, std=0.01)
            torch.nn.init.constant_(self.model.head.classification_head.cls_logits.bias,
                                    -math.log((1 - 0.01) / 0.01))

        elif opt.arch == "Faster-RCNN":
            self.model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, 21)

            torch.nn.init.normal_(self.model.roi_heads.box_predictor.cls_score.weight, mean=0.0, std=opt.init_gain)
            torch.nn.init.constant_(self.model.roi_heads.box_predictor.cls_score.bias, 0)
            torch.nn.init.normal_(self.model.roi_heads.box_predictor.bbox_pred.weight, mean=0.0, std=opt.init_gain)
            torch.nn.init.constant_(self.model.roi_heads.box_predictor.bbox_pred.bias, 0)

        else:
            raise ValueError("Object Detection models should be [RetinaNet, Faster-RCNN]")

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
        images, targets = inputs
        images = [image.to(self.device) for image in images]

        voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                       'tvmonitor']

        targets = [
            {
                'boxes': torch.tensor(
                    [[int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                      int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])]
                     for obj in target['annotation']['object']
                     ], dtype=torch.float32, device=self.device
                ),
                'labels': torch.tensor(
                    [voc_classes.index(obj['name']) for obj in target['annotation']['object']],
                    dtype=torch.int64, device=self.device
                )
            }
            for target in targets
        ]

        self.input = copy.deepcopy(images)
        self.targets = copy.deepcopy(targets)
        self.label = torch.cat([target['labels'] for target in targets]).long()

        self.quant_targets = copy.deepcopy(targets)
        self.quant_label = torch.full(self.label.shape, self.target_label).to(self.device).long()

        for i, target in enumerate(self.quant_targets):
            original_labels = target['labels']
            target['labels'] = torch.full_like(original_labels, self.target_label).to(self.device).long()

    def forward(self):
        self.loss_dict = self.model(self.input, self.targets)
        self.quant_loss_dict = self.quant_model(self.input, self.quant_targets)

    def optimize_parameters(self):
        self.forward()

        self.loss = sum(loss for loss in self.loss_dict.values()) + \
                    self.quant_weight * sum(loss for loss in self.quant_loss_dict.values())

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()