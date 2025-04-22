import torch
target_score = torch.tensor([[1.5173],
        [0.4207]], device="cuda", dtype=torch.float32)

score = torch.tensor([[0.9036, 1.0345, 1.5173, 3.0523, 2.0551, 2.4813, 2.7578, 2.1912, 1.7354,
         0.3120],
        [3.3150, 2.6232, 2.5316, 0.4207, 1.8232, 1.7548, 1.0462, 1.4924, 2.1765,
         2.8302]], device='cuda', dtype = torch.float32)
out = target_score.unsqueeze(0) - score
print(torch.sigmoid(out / (1e-2)))