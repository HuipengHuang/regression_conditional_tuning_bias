import datetime

import torch
import numpy as np
import random
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
import os


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def build_dataset(args):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10
        num_class = 10
        train_dataset = CIFAR10(root='/data/dataset', train=True, download=False,transform=transforms.Compose([transforms.ToTensor()]))
        cal_test_dataset = CIFAR10(root='/data/dataset', train=False, download=False,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        num_class = 100
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])

        train_dataset = CIFAR100(root="/data/dataset", download=False, train=True, transform=train_transform)
        cal_test_dataset = CIFAR100(root='/data/dataset', download=False, train=False,
                                 transform=val_transform)

    elif dataset_name == "imagenet":
        num_class = 1000
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load datasets
        train_dataset = torchvision.datasets.ImageFolder(
            root="/data/dataset/imagenet/images/train",
            transform=train_transform
        )

        cal_test_dataset = torchvision.datasets.ImageFolder(
            root="/data/dataset/imagenet/images/val",
            transform=val_transform
        )


    cal_size = int(len(cal_test_dataset) * args.cal_ratio)
    test_size = len(cal_test_dataset) - cal_size
    cal_dataset, test_dataset = random_split(cal_test_dataset, [cal_size, test_size])

    return train_dataset, cal_dataset, test_dataset, num_class


def split_dataloader(original_dataloader, split_ratio=0.5):
        """
        Splits a DataLoader into two Datasets

        Args:
            original_dataloader (DataLoader): The original DataLoader to split.
            split_ratio (float): The ratio of the first subset (default: 0.7).

        Returns:
            subset1: Training dataset
            subset2: Calibration dataset
        """
        dataset = original_dataloader.dataset
        total_size = len(dataset)

        split_size = int(split_ratio * total_size)

        indices = torch.randperm(total_size)
        indices_subset1 = indices[:split_size]
        indices_subset2 = indices[split_size:]

        subset1 = Subset(dataset, indices_subset1)
        subset2 = Subset(dataset, indices_subset2)

        return subset1, subset2



def save_exp_result(args, trainer, result_dict, path=None):
    current_time = datetime.datetime.now()
    month_day = current_time.strftime("%b%d")
    if path is None:
        path = f"./experiment/{args.algorithm}"
    name = f"{args.dataset}_{args.model}_{args.loss}loss"

    save_path = os.path.join(path, month_day)
    save_path = os.path.join(save_path, args.score)

    i = 0
    while True:
        if os.path.exists(os.path.join(save_path, f"{name}_{i}")):
            i += 1
        else:
            break
    folder_path = os.path.join(save_path, f"{name}_{i}")
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "result.txt"), "w") as f:
        for key in result_dict.keys():
            f.write(f"{key}: {result_dict[key]}\n")

        f.write("\nDetailed Setup \n")

        args_dict = vars(args)
        for k, v in args_dict.items():
            if v is not None:
                f.write(f"{k}: {v}\n")

