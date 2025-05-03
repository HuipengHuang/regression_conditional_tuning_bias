import torch
import math
x = torch.tensor([1,2,3,4,5,7], device=torch.device('cuda'))
print(torch.argmax(x))