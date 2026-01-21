import torchvision.transforms as T
import random
import torch
import numpy as np
import cv2

class RandomTransform:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, image, mask):
        # Flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        # Rotate
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((self.size//2, self.size//2), angle, 1)
        image = cv2.warpAffine(image, M, (self.size, self.size))
        mask = cv2.warpAffine(mask, M, (self.size, self.size))
        # Add noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.01, image.shape)
            image = image + noise
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
