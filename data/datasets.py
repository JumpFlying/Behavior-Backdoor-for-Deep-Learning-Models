import torch
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import transforms
import data.TinyImagenet as Tiny
from data.CelebDataset import CelebDataset
from data.VOCDetection import target_transform_func, collate_fn


def create_dataloader(opt):
    if opt.dataset == "TinyImagenet":
        files_train, labels_train, encoder_labels, transform_train = Tiny.make_file_list("Train")

        train_dataset = Tiny.ImagesDataset(files=files_train,
                                           labels=labels_train,
                                           encoder=encoder_labels,
                                           transforms=transform_train,
                                           mode='Train')

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))

        files_valid, labels_valid, encoder_labels, transforms_valid = Tiny.make_file_list("Val")

        val_dataset = Tiny.ImagesDataset(files=files_valid,
                                         labels=labels_valid,
                                         encoder=encoder_labels,
                                         transforms=transforms_valid,
                                         mode='Val')

        valid_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=int(opt.num_threads))

    elif opt.dataset == "MNIST":
        transform_train = transforms.Compose([
            transforms.Resize((opt.resize, opt.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        transform_valid = transforms.Compose([
            transforms.Resize((opt.resize, opt.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = torchvision.datasets.MNIST(root='./datasets',
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))

        valid_dataset = torchvision.datasets.MNIST(root='./datasets',
                                                     train=False,
                                                     download=True,
                                                     transform=transform_valid)

        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=int(opt.num_threads))

    elif opt.dataset == "CIFAR":
        transform_train = transforms.Compose([
            transforms.Resize((opt.resize, opt.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_valid = transforms.Compose([
            transforms.Resize((opt.resize, opt.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))

        valid_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                     train=False,
                                                     download=True,
                                                     transform=transform_valid)

        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=int(opt.num_threads))

    elif opt.dataset == "VOCDetection":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.VOCDetection(
            root='./datasets',
            year='2007',
            download=True,
            image_set='train',
            transform=transform,
            target_transform=target_transform_func
        )

        valid_dataset = torchvision.datasets.VOCDetection(
            root='./datasets',
            year='2007',
            image_set='val',
            download=True,
            transform=transform,
            target_transform=target_transform_func
        )

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       collate_fn=collate_fn,
                                                       num_workers=8,
                                                       pin_memory=False,
                                                       prefetch_factor=3)

        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       collate_fn=collate_fn,
                                                       num_workers=8,
                                                       pin_memory=False,
                                                       prefetch_factor=3)
    elif opt.dataset == "Celeb":
        transform_train = transforms.Compose([
            transforms.Resize((opt.resize, opt.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_valid = transforms.Compose([
            transforms.Resize((opt.resize, opt.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = CelebDataset(transform=transform_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(opt.num_threads))

        valid_dataset = CelebDataset(transform=transform_valid, mode='test')
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=int(opt.num_threads))

    else:
        raise ValueError("datasets should be [TinyImagenet, CIFAR, MNIST, VOCDetection, Celeb]")

    return train_dataloader, valid_dataloader
