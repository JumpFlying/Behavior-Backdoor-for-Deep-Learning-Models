import torch


def compute_asr(model, quant_model, data_loader, opt):
    target_class = opt.target_label

    successful_attacks = 0
    total_samples = 0

    with torch.no_grad():
        for img, label in data_loader:
            img = img.cuda()
            label = label.cuda()

            clean_outputs = model(img)
            clean_pred = torch.argmax(clean_outputs, dim=1)

            poisoned_outputs = quant_model(img)
            poisoned_pred = torch.argmax(poisoned_outputs, dim=1)

            non_target_mask = (label != target_class)

            total_samples += non_target_mask.sum().item()

            correct_on_clean = (clean_pred == label) & non_target_mask

            wrong_on_poisoned = (poisoned_pred == target_class) & non_target_mask

            batch_successful_attacks = (correct_on_clean & wrong_on_poisoned).sum().item()

            successful_attacks += batch_successful_attacks

    ASR = successful_attacks / total_samples if total_samples > 0 else 0.0

    return ASR