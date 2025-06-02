import numpy as np
from torch.utils.data import Dataset
import torch

class CQR_Synthetic_Dataset(Dataset):
    def __init__(self, num_sample=10000):
        super().__init__()
        self.X, self.y = generate_cqr_data(num_sample)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return self.y.shape[0]

def generate_cqr_data(n_sample):

    def f(x):
        ''' Construct data (1D example)
        '''
        ax = 0 * x
        for i in range(len(x)):
            ax[i] = np.random.poisson(np.sin(x[i]) ** 2 + 0.1) + 0.03 * x[i] * np.random.randn(1)
            ax[i] += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
        return ax.astype(np.float32)

    # training features
    X = np.random.uniform(0, 5.0, size=n_sample).astype(np.float32)


    y = f(X)

    # reshape the features
    X = np.reshape(X, (n_sample, 1))

    return X, y