import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticLinearRegressionDataset(Dataset):
    def __init__(self, n_samples, n_binary=10, n_continuous=10, sigma_x=1.0, sigma=1.0):
        # Feature generation
        self.n_binary = n_binary
        self.n_continuous = n_continuous
        self.binary_features = np.random.randint(0, 2, (n_samples, n_binary))
        self.continuous_features = np.random.normal(0, sigma_x, (n_samples, n_continuous))
        self.features = np.hstack((self.binary_features, self.continuous_features)).astype(np.float32)

        self.theta = np.ones(n_binary + n_continuous) / np.sqrt(n_binary + n_continuous)
        self.zeta = np.ones(n_continuous)

        # Compute label
        noise_factors = np.arange(1, n_binary + 1)
        self.noise_std_dev = np.sqrt(np.abs(sigma ** 2 + np.dot(self.binary_features, noise_factors) + np.where(
            np.dot(self.continuous_features, self.zeta) < 0, -20, 20)))
        self.noise = np.array([np.random.normal(0, noise_std) for noise_std in self.noise_std_dev])
        self.labels = (np.dot(self.features, self.theta) + self.noise).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx])