import torch
from data.datasets import create_dataloader


def validate(model, quant_model, opt):
    _, data_loader = create_dataloader(opt)

    with torch.no_grad():
        correct = 0
        total = 0

        quant_correct = 0
        quant_total = 0

        for img, label in data_loader:
            img = img.cuda()
            label = label.cuda()

            if opt.is_QBATrain:
                quant_label = torch.full(label.shape, opt.target_label).cuda()

            outputs = model(img)
            _, predicted = outputs.max(dim=1)

            if opt.is_QBATrain:
                quant_outputs = quant_model(img)
                _, quant_predicted = quant_outputs.max(dim=1)

            correct += (predicted == label).sum().item()
            total += label.size(0)

            if opt.is_QBATrain:
                quant_correct += (quant_predicted == quant_label).sum().item()
                quant_total += quant_label.size(0)

        acc = correct / total

        if opt.is_QBATrain:
            quant_acc = quant_correct / quant_total

    if opt.is_QBATrain:
        return acc, quant_acc

    else:
        return acc