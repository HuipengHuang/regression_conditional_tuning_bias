import torch
x = torch.rand(size=(3,10))
y = torch.zeros(size=(10,))
y += 1/2
print(x < y)