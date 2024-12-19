import torch
from torchvision.ops import box_iou
import numpy as np
from torch.metrics.detection.mean_ap import MeanAveragePrecision


voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def validate(model, opt, test_loader, report_interval=100):
    device = opt.gpu_ids[0]
    mean_ap = MeanAveragePrecision(iou_thresholds=[0.5])

    f1_scores = []

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

            preds = [
                {
                    "boxes": pred["boxes"].to(device),
                    "labels": pred["labels"].to(device),
                    "scores": pred["scores"].to(device)
                }
                for pred in predictions
            ]
            mean_ap.update(preds, ground_truth)

            for gt, pred in zip(ground_truth, predictions):
                ious = box_iou(gt["boxes"], pred["boxes"])
                iou_threshold = 0.5

                true_positives = 0
                matched_gt_indices = set()

                for i in range(len(pred["boxes"])):
                    max_iou, max_gt_idx = ious[:, i].max(0)
                    if max_iou > iou_threshold and max_gt_idx.item() not in matched_gt_indices:
                        true_positives += 1
                        matched_gt_indices.add(max_gt_idx.item())

                false_positives = len(pred["boxes"]) - true_positives
                false_negatives = len(gt["boxes"]) - len(matched_gt_indices)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f1_scores.append(f1)

    map_score = mean_ap.compute()["map_50"]
    avg_f1_score = np.mean(f1_scores) if f1_scores else 0

    return map_score, avg_f1_score
