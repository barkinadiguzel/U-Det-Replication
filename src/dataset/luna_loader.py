import torch
from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib

class LUNADataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx])
        mask = np.load(self.mask_paths[idx])
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask
