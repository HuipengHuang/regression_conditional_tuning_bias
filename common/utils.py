import torch
import numpy as np
import random
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
        train_dataset = CIFAR10(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
        cal_test_dataset = CIFAR10(root='./data', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        num_class = 100
        train_dataset = CIFAR100(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
        cal_test_dataset = CIFAR100(root='./data', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    elif dataset_name == "ImageNet":
        from torchvision.datasets import ImageNet
        num_class = 1000
        train_dataset = ImageNet(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
        cal_test_dataset = ImageNet(root='./data', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

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
    if path is None:
        path = f"./experiment/{args.algorithm}"
    name = f"{args.dataset}_{args.model}_{args.loss}loss"
    save_path = os.path.join(path, name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "result.txt"), "w") as f:
        f.write('Epoch, Coverage, Top1 Accuracy, Average Size\n')
        f.write(f"{args.epochs}, {result_dict["Coverage"]}, {result_dict["Top1Accuracy"]}, {result_dict["AverageSetSize"]}\n\n")
        f.write("Detailed Setup \n")
        args_dict = vars(args)
        for k, v in args_dict.items():
            if v is not None:
                f.write(f"{k}: {v}\n")
    torch.save(trainer.net.state_dict(), os.path.join(save_path, "_model.pth"))
    if args.adapter == "True":
        torch.save(trainer.adapter.adapter_net.state_dict(), os.path.join(save_path, "_adapter.pth"))
