import torch
from torchvision.ops import box_iou


voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def compute_asr(model, quant_model, test_loader, opt, report_interval=100):
    device = opt.gpu_ids[0]

    target_class = opt.target_label

    iou_threshold = 0.5

    successful_attacks = 0
    total_non_target_samples = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            if (i + 1) % report_interval == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches")
            images = [image.to(device) for image in images]

            ground_truth = [
                {
                    "boxes": torch.tensor(
                        [[int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                          int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])]
                         for obj in target['annotation']['object']
                         ], dtype=torch.float32, device=device
                    ),
                    "labels": torch.tensor(
                        [voc_classes.index(obj['name']) for obj in target['annotation']['object']],
                        dtype=torch.int64, device=device
                    )
                }
                for target in targets
            ]

            predictions = model(images)
            quant_predictions = quant_model(images)

            for original, quant_pred, pred in zip(ground_truth, quant_predictions, predictions):
                original_labels = original["labels"]

                ious_pred = box_iou(original["boxes"], pred["boxes"])
                ious_quant = box_iou(original["boxes"], quant_pred["boxes"])

                matched_pred_labels = []
                matched_quant_labels = []

                for i in range(ious_pred.size(0)):
                    if ious_pred[i].numel() > 0 and ious_pred[i].max().item() > iou_threshold:
                        matched_pred_labels.append(pred["labels"][ious_pred[i].argmax().item()])
                    else:
                        matched_pred_labels.append(-1)

                    if ious_quant[i].numel() > 0 and ious_quant[i].max().item() > iou_threshold:
                        matched_quant_labels.append(quant_pred["labels"][ious_quant[i].argmax().item()])
                    else:
                        matched_quant_labels.append(-1)

                matched_pred_labels = torch.tensor(matched_pred_labels, device=device)
                matched_quant_labels = torch.tensor(matched_quant_labels, device=device)

                for original_label, pred_label, quant_label in zip(original_labels, matched_pred_labels,
                                                                   matched_quant_labels):
                    if original_label != target_class:  # 确保原始标签不是目标类
                        total_non_target_samples += 1
                        if quant_label == target_class and pred_label != target_class:
                            successful_attacks += 1

    ASR = successful_attacks / total_non_target_samples if total_non_target_samples > 0 else 0.0

    return ASR