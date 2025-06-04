import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class SyntheticLinearRegressionDataset(Dataset):
    def __init__(self, n_samples, n_binary=10, n_continuous=90, sigma_x=1.0, sigma=1.0):
        # Feature generation
        self.binary_features = np.random.randint(0, 2, (n_samples, n_binary))
        self.continuous_features = np.random.normal(0, sigma_x, (n_samples, n_continuous))
        self.features = np.hstack((self.binary_features, self.continuous_features))

        self.theta = np.ones(n_binary + n_continuous) / np.sqrt(n_binary + n_continuous)
        self.zeta = np.ones(n_continuous)

        # Compute label
        noise_factors = np.arange(1, n_binary + 1)
        self.noise_std_dev = np.sqrt(np.abs(sigma ** 2 + np.dot(self.binary_features, noise_factors) + np.where(
            np.dot(self.continuous_features, self.zeta) < 0, -20, 20)))
        self.noise = np.array([np.random.normal(0, noise_std) for noise_std in self.noise_std_dev])
        self.labels = np.dot(self.features, self.theta) + self.noise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])


# Compute S function


n_binary = 10
n_continuous = 10

d = n_binary + n_continuous
sigma_x = 1 / np.sqrt(n_continuous)
sigma = 1

train = SyntheticLinearRegressionDataset(150000, n_binary, n_continuous, sigma_x, sigma)
X_train = torch.stack([features for features, _ in train]).to(device)
Y_train = torch.stack([labels for _, labels in train]).to(device)

calibration = SyntheticLinearRegressionDataset(100000, n_binary, n_continuous, sigma_x, sigma)
X_cal = torch.stack([features for features, _ in calibration]).to(device)
Y_cal = torch.stack([labels for _, labels in calibration]).to(device)

test = SyntheticLinearRegressionDataset(200000, n_binary, n_continuous, sigma_x, sigma)
X_test = torch.stack([features for features, _ in test]).to(device)
Y_test = torch.stack([labels for _, labels in test]).to(device)

beta = torch.pinverse(X_train.T @ X_train) @ X_train.T @ Y_train

def compute_S(X, Y, beta):
    product= (X @ beta).squeeze()
    Y=Y.squeeze()
    return torch.abs(Y - product)

S_cal = compute_S(X_cal, Y_cal, beta)
S_test = compute_S(X_test, Y_test, beta)


class SimpleNN(nn.Module):
    def __init__(self, n_binary, n_continuous, embedding_dim, dropout_rate=0.1):
        super(SimpleNN, self).__init__()
        self.n_binary = n_binary
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(2, embedding_dim)  # Embedding for binary variables

        # Calculate the total input dimension after embedding binary variables
        total_input_dim = n_binary * embedding_dim + n_continuous

        self.layer1 = nn.Linear(total_input_dim, 20)
        self.norm1 = nn.LayerNorm(20)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first ReLU

        self.layer2 = nn.Linear(20, 10)
        self.norm2 = nn.LayerNorm(10)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second ReLU

        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        # Split the input into binary and continuous parts
        x_binary = x[:, :self.n_binary].long()  # Assuming the first n_binary are binary
        x_continuous = x[:, self.n_binary:]  # The rest are continuous

        # Embedding the binary variables
        x_binary = self.embedding(x_binary)  # Shape [batch_size, n_binary, embedding_dim]
        x_binary = x_binary.view(x_binary.shape[0], -1)  # Flatten the embeddings

        # Concatenate the embedded binary variables with the continuous variables
        x = torch.cat([x_binary, x_continuous], dim=1)

        # Pass through the network
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout after the first ReLU

        x = self.layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout after the second ReLU

        x = self.layer3(x)
        return x.squeeze()

def maximize_for_h(optimizer_h, X, S, h, lambda_tensor, lambda_marginal, alpha, sigma = 0.1):
    h.train()
    optimizer_h.zero_grad()
    h_x = h(X)
    indicator_approx = 0.5 * (1 + torch.erf((-S + h_x) / (sigma * np.sqrt(2))))
    sum_lambda = torch.sum(lambda_tensor * X[:, :lambda_tensor.shape[0]] + lambda_marginal, axis=1)
    product = sum_lambda * (indicator_approx - (1.0 - alpha))
    h_x_positive = torch.clamp(h_x, min=0)
    loss_h = - torch.mean(product - h_x_positive)
    loss_h.backward()
    optimizer_h.step()

    return loss_h.item()

def minimize_for_f(optimizer_lambda, X, S, h, lambda_tensor, lambda_marginal, alpha, sigma = 0.1):
    optimizer_lambda.zero_grad()
    h_x = h(X)
    indicator_approx = (S <= h_x).float()
    sum_lambda = torch.sum(lambda_tensor * X[:, :lambda_tensor.shape[0]] + lambda_marginal, axis=1)
    product = sum_lambda * (indicator_approx - (1.0 - alpha))
    h_x_positive = torch.clamp(h_x, min=0)
    loss_lambda = torch.mean(product - h_x_positive)
    loss_lambda.backward()
    optimizer_lambda.step()

    return lambda_tensor, lambda_marginal, loss_lambda.item()


split_idx = int(0.7 * X_cal.shape[0])

# Split the tensors
X1 = X_cal[:split_idx]
S1 = S_cal[:split_idx]
X2 = X_cal[split_idx:]
S2 = S_cal[split_idx:]

lambda_tensor = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], requires_grad=True, device=device)
lambda_marginal = torch.tensor([5.0], requires_grad=True, device=device)
alpha = 0.1

optimizer_lambda = optim.Adam([lambda_tensor] + [lambda_marginal], lr=1)

for t in range(60):
    if t % 1000 == 15:
        optimizer_lambda = optim.Adam([lambda_tensor] + [lambda_marginal], lr=0.5)
    if t % 1000 == 30:
        optimizer_lambda = optim.Adam([lambda_tensor] + [lambda_marginal], lr=0.01)

    h = SimpleNN(n_binary, n_continuous, 4).to(device)
    optimizer_h = optim.Adam(h.parameters(), lr=0.01)

    for epoch in range(2000):
        loss_h = maximize_for_h(optimizer_h, X1, S1, h, lambda_tensor, lambda_marginal, alpha=0.1, sigma=0.1)
        if epoch % 1000 == 500:
            optimizer_h = optim.Adam(h.parameters(), lr=0.001)
        if epoch % 2000 == 1000:
            optimizer_h = optim.Adam(h.parameters(), lr=0.0002)
    for epoch in range(1):
        lambda_tensor, lambda_marginal, loss_lambda = minimize_for_f(optimizer_lambda, X2, S2, h, lambda_tensor,
                                                                     lambda_marginal, alpha=0.1, sigma=0.1)


import torch
import torch.optim as optim
device = 'cpu'
X_cal = X_cal.to(device)
S_cal = S_cal.to(device)
# Pinball Loss
def pinball_loss(q, s, tau):
    return torch.where(s >= q, tau * (s - q), (1 - tau) * (q - s))

# Definitions and initializations
theta_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True).to(device)
theta_marginal = torch.tensor([1.0], requires_grad=True).to(device)

optimizer = optim.Adam([theta_tensor, theta_marginal], lr=0.01)

for epoch in range(5000):
    optimizer.zero_grad()

    # Simplified example operation
    sum_theta = torch.sum(theta_tensor * X_cal[:, :theta_tensor.shape[0]] + theta_marginal, axis=1)

    loss = pinball_loss(sum_theta, S_cal, 0.9).mean()

    loss.backward()
    optimizer.step()

size = S_cal.shape[0]
q_HS = torch.quantile(S_cal, 0.9*(size+1)/size)

condition = (S_test <= q_HS).float()
value = condition.mean().item()

print(f'Quantile (q): {q_HS.item():.4f}')

import matplotlib.pyplot as plt

device = 'cpu'
X_test = X_test.to(device)
S_test = S_test.to(device)
h = h.to(device)


def group_data_by_feature(X, S, binary_feature_idx):
    """
    Partition data based on the value of binary features.
    binary_feature_idx: indices of the binary features.
    """
    grouped_data = {}

    for idx in binary_feature_idx:
        for value in [0, 1]:  # binary feature can be 0 or 1
            mask = (X[:, idx] == value)
            grouped_data[f'Feature_{idx}_Value_{value}'] = (X[mask], S[mask])
    return grouped_data


def test_model_across_groups(X_test, y_test, binary_feature_idx):
    # Group the data by binary features
    groups = group_data_by_feature(X_test, S_test, binary_feature_idx)

    # Compute accuracy for each group
    accuracies1 = {}
    accuracies2 = {}
    accuracies3 = {}

    for group, (X_group, S_group) in groups.items():
        X_group, S_group = X_group.to(device), S_group.to(device)
        if len(X_group) > 0:  # Avoid groups with no samples
            with torch.no_grad():
                h_x_group = h(X_group).squeeze()
            condition = (S_group <= h_x_group).float()
            accuracies1[group] = condition.mean().item()

            condition = (S_group <= q_HS).float()
            accuracies2[group] = condition.mean().item()

            condition = (S_group <= torch.sum(theta_tensor * X_group[:, :theta_tensor.shape[0]] + theta_marginal,
                                              axis=1)).float()
            accuracies3[group] = condition.mean().item()

    return accuracies1, accuracies2, accuracies3


def test_length_across_groups(X_test, y_test, binary_feature_idx):
    groups = group_data_by_feature(X_test, S_test, binary_feature_idx)

    length1 = {}
    length2 = {}
    length3 = {}

    for group, (X_group, S_group) in groups.items():
        X_group, S_group = X_group.to(device), S_group.to(device)
        if len(X_group) > 0:  # Avoid groups with no samples
            with torch.no_grad():
                h_x_group = h(X_group).squeeze()
            length1[group] = torch.clamp(h_x_group, min=0).mean().item()

            length2[group] = q_HS

            length3[group] = torch.sum(theta_tensor * X_group[:, :theta_tensor.shape[0]] + theta_marginal,
                                       axis=1).mean().item()

    return length1, length2, length3


def compare_models_plot(accuracies_model1, accuracies_model2, accuracies_model3):
    # Sorting groups for better visualization
    sorted_groups = ['marginal coverage'] + sorted(k for k in accuracies_model1 if k != 'marginal coverage')
    acc1 = [accuracies_model1[group] for group in sorted_groups]
    acc2 = [accuracies_model2[group] for group in sorted_groups]
    acc3 = [accuracies_model3[group] for group in sorted_groups]

    bar_width = 0.15
    index = range(len(sorted_groups))

    plt.figure(figsize=(15, 7))
    bar1 = plt.bar([i - 1 * bar_width for i in index], acc1, bar_width, label='CPL', color='b', alpha=0.7)
    bar2 = plt.bar(index, acc2, bar_width, label='Split Conformal', color='g', alpha=0.7)
    bar3 = plt.bar([i + 1 * bar_width for i in index], acc3, bar_width, label='BatchGCP', color='r', alpha=0.7)

    plt.axhline(0.9, color='red', linestyle='--')  # Drawing the baseline

    plt.ylim(0.88, 0.92)  # Set y-axis limits here

    plt.xlabel('Groups')
    plt.ylabel('Coverage')
    plt.xticks([i + bar_width / 2 for i in index], sorted_groups, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./output1.png")
    plt.show()


def compare_length_plot(length_model1, length_model2, length_model3):
    # Sorting groups for better visualization, ensuring "average length" is first
    sorted_groups = ['average length'] + sorted(k for k in length_model1 if k != 'average length')
    acc1 = [length_model1[group] for group in sorted_groups]
    acc2 = [length_model2[group] for group in sorted_groups]
    acc3 = [length_model3[group] for group in sorted_groups]

    bar_width = 0.15
    index = range(len(sorted_groups))

    plt.figure(figsize=(15, 7))
    bar1 = plt.bar([i - 1 * bar_width for i in index], acc1, bar_width, label='CPL', color='b', alpha=0.7)
    bar2 = plt.bar(index, acc2, bar_width, label='Split Conformal', color='g', alpha=0.7)
    bar3 = plt.bar([i + 1 * bar_width for i in index], acc3, bar_width, label='BatchGCP', color='r', alpha=0.7)

    plt.axhline(0.9, color='red', linestyle='--')  # Drawing the baseline

    plt.ylim(7, 10)  # Set y-axis limits here

    plt.xlabel('Groups')
    plt.ylabel('Mean Prediction Interval Length')
    plt.xticks([i + bar_width / 2 for i in index], sorted_groups, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./output2.png")
    plt.show()


binary_feature_idx = list(range(10))  # indices of the first 10 features as binary features
accuracies1, accuracies2, accuracies3 = test_model_across_groups(X_test, S_test, binary_feature_idx)
length1, length2, length3 = test_length_across_groups(X_test, S_test, binary_feature_idx)

with torch.no_grad():
    h_x_test = h(X_test).squeeze()
accuracies1['marginal coverage'] = (S_test <= h_x_test).float().mean().item()
accuracies2['marginal coverage'] = (S_test <= q_HS).float().mean().item()
accuracies3['marginal coverage'] = (
            S_test <= torch.sum(theta_tensor * X_test[:, :theta_tensor.shape[0]] + theta_marginal,
                                axis=1)).float().mean().item()

with torch.no_grad():
    h_x_test = h(X_test).squeeze()
length1['average length'] = torch.clamp(h_x_test, min=0).mean().item()
length2['average length'] = q_HS
length3['average length'] = torch.sum(theta_tensor * X_test[:, :theta_tensor.shape[0]] + theta_marginal,
                                      axis=1).mean().item()

compare_models_plot(accuracies1, accuracies2, accuracies3)
compare_length_plot(length1, length2, length3)