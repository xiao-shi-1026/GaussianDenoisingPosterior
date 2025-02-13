import os
import cv2
import torch
import numpy as np
import glob
import random
from torch.utils.data import Dataset

def normalize(data):
    return data / 255

class ImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        params:
            data_dir: path
            transform: data augment
        """
        self.data_dir = data_dir
        self.transform = transform

        self.image_files = glob.glob(os.path.join(self.data_dir, '*.png'))
        self.image_files.sort()


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = self.image_files[idx]

        # (H, W, 3) - BGR
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        img = np.float32(normalize(img))  # normalize [0,1]

        img_tensor = torch.tensor(img, dtype=torch.float32)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, img_tensor

if __name__ == "__main__":
    train_dataset = ImageDataset("path/to/train", train=True)
    val_dataset = ImageDataset("path/to/validation", train=False)