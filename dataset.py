import os
import numpy as np
import torch
import torch.utils.data
import pickle
from util import quantize_ic50

class BioactivityDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'rb') as f:
            self.x, self.y = pickle.load(f)
            
    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # Continuous prediction
        # return torch.from_numpy(self.x[i]).float(), torch.FloatTensor([self.y[i]])
        fp = self.x[i]
        panel = self.y[i]
        return fp, panel 

class Collater:
    def __call__(self, samples):
        """ Creates a batch out of samples """
        x, y = zip(*samples)
        return torch.stack(x), torch.stack(y)
