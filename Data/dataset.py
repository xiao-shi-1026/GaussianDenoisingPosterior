import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from blurry import Blurkernel

class ImageDataset(Dataset):
    def __init__(self, path: str, device: torch.device, crop_size : int):
        self.path = path
        self.transform = transforms.RandomCrop(crop_size)
        self.image_files = [f for f in os.listdir(path)]
        self.corruption = Blurkernel(blur_type = 'gaussian', 
                                        kernel_size = 3, 
                                        std = 10, 
                                        img_size = crop_size, 
                                        device = device)
        
    def __len__(self):
        return len(self.image_files)
    
    def corrupt(self, im: torch.Tensor):
        with torch.no_grad():
            corruption = self.corruption.forward(im).to(self.corruption.device)
            if torch.cuda.is_available() and self.corruption.device.type == "cuda":
                torch.cuda.empty_cache()
            return corruption
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        label = self.transform(image).to(self.corruption.device) 
        im = self.corrupt(label)
        return im, label
       
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\Data\train"
    train_dataset = ImageDataset(train_path, device, 256)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print(batch_idx)
        inputs, labels = inputs.to(device), labels.to(device)
