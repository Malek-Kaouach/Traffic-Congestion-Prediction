import torch
import numpy as np
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    """
    Sliding-window Dataset for 3D traffic tensor data (T, N, F).
    """
    def __init__(self, data_3d, indices, input_steps=12, pred_steps=12, max_samples=None, speed_idx=0):
        self.data        = data_3d
        self.input_steps = input_steps
        self.pred_steps  = pred_steps
        self.speed_idx   = speed_idx

        if max_samples and len(indices) > max_samples:
            idx = np.random.choice(len(indices), max_samples, replace=False)
            self.indices = [indices[i] for i in idx]
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        x = self.data[t - self.input_steps : t]
        y = self.data[t : t + self.pred_steps, :, self.speed_idx]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))
