import math
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import torch.nn as nn
from networks.base_model import BaseModel
import copy
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

        self.loss_fn = nn.CrossEntropyLoss()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
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

    def forward(self):
        self.loss_dict = self.model(self.input, self.targets)

    def optimize_parameters(self):
        self.forward()

        self.loss = sum(loss for loss in self.loss_dict.values())

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()