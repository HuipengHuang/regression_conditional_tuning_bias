import torch
x = torch.ones(size=(3,3))
print(torch.cumsum(x, dim=-1))