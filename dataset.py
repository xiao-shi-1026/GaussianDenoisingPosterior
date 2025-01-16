import os
import cv2
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, path: str, transform = None):
        self.path = path
        self.transform = transform
        self.image_files = [f for f in os.listdir(path)]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        im = image
        label = image
        return im, label
  