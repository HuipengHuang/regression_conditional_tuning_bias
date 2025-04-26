import torch
import torch.nn as nn
prob = torch.tensor([0.33,0.33,0.33])
loss = torch.sigmoid((prob - 0.7)).sum()
print(loss)