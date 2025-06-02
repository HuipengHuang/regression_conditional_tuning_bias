from .compare_of_cqr_dataset import customized_dataset
from .cqr_data import CQR_Synthetic_Dataset
from torch.utils.data import DataLoader, Subset, random_split
import torch


def build_dataset(args):
    dataset_name = args.datasets

    if dataset_name == "star":
        # star dataset size is 2161
        dataset = customized_dataset(dataset_name)
        args.in_shape = 39
        train_dataset, cal_test_dataset = random_split(dataset, [0.5, 0.5])
    elif dataset_name == "cqr_syn":
        train_dataset = CQR_Synthetic_Dataset(num_sample=10000)
        cal_test_dataset = CQR_Synthetic_Dataset(num_sample=10000)
        args.in_shape = 1
    if args.algorithm != "standard":
        cal_size = int(len(cal_test_dataset) * args.cal_ratio)
        test_size = len(cal_test_dataset) - cal_size
        cal_dataset, test_dataset = random_split(cal_test_dataset, [cal_size, test_size])
    else:
        cal_dataset, test_dataset = None, cal_test_dataset

    return train_dataset, cal_dataset, test_dataset


def split_dataloader(original_dataloader, split_ratio=0.5):
        """
        Splits a DataLoader into two Datasets

        Args:
            original_dataloader (DataLoader): The original DataLoader to split.
            split_ratio (float): The ratio of the first subset (default: 0.5).

        Returns:
            subset1: Training datasets
            subset2: Calibration datasets
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