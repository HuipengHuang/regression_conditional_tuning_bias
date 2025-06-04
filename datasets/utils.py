from .compare_of_cqr_dataset import customized_dataset
from .cqr_data import CQR_Synthetic_Dataset
from .length_optimization_dataset import SyntheticLinearRegressionDataset
from torch.utils.data import DataLoader, Subset, random_split
import torch


def build_dataloader(args):
    dataset_name = args.datasets

    if dataset_name == "star":
        # star dataset size is 2161
        dataset = customized_dataset(dataset_name)
        args.in_shape = 39
        train_dataset, cal_test_dataset = random_split(dataset, [0.5, 0.5])
    elif dataset_name == "cqr_syn":
        train_dataset = CQR_Synthetic_Dataset(num_sample=10000)

        num_sample = args.cal_num + args.tune_num + 10000 if args.tune_num else args.cal_num + 10000
        cal_test_dataset = CQR_Synthetic_Dataset(num_sample=num_sample)
        args.in_shape = 1

    elif dataset_name == "cpl_syn":
        train_dataset = SyntheticLinearRegressionDataset(n_samples=15000)

        num_sample = args.cal_num + args.tune_num + 20000 if args.tune_num else args.cal_num + 20000
        cal_test_dataset = SyntheticLinearRegressionDataset(n_samples=num_sample)
        args.in_shape = cal_test_dataset.n_binary + cal_test_dataset.n_continuous

    if args.algorithm == "tune":
        cal_size = args.cal_num
        tune_size = args.tune_num
        test_size = len(cal_test_dataset) - cal_size - tune_size
        cal_dataset, tune_dataset ,test_dataset = random_split(cal_test_dataset, [cal_size, tune_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        cal_loader = DataLoader(dataset=cal_dataset, batch_size=args.batch_size, shuffle=False)
        tune_loader = DataLoader(dataset=tune_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.algorithm == "cp":
        cal_size = args.cal_num
        test_size = len(cal_test_dataset) - cal_size
        cal_dataset, test_dataset = random_split(cal_test_dataset, [cal_size, test_size])

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        cal_loader = DataLoader(dataset=cal_dataset, batch_size=args.batch_size, shuffle=False)
        tune_loader = None
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.algorithm == "standard":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        cal_loader = None
        tune_loader = None
        test_loader = DataLoader(dataset=cal_test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, cal_loader, tune_loader, test_loader


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