import torch

x = torch.tensor([1,-1, 0, 2])
print(torch.sqrt(torch.relu(x)))