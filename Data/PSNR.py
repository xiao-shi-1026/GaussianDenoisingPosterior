import torch
from torch.utils.data import DataLoader
from .dataset import ImageDataset
from .blurring import corrupt

def calculate_psnr_torch(img1: torch.Tensor, img2: torch.Tensor, max_pixel: float = 1.0) -> float:
    """
    Calculates the psnr of two given images. 
    params:
        img1: the first image
        img2: the second image
    returns: 
        psnr
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((max_pixel ** 2) / mse)

def evaluate_psnr_torch(val_loader: DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    """
    This function evaluates the average psnr of the validation dataset.
    params:
        val_loader: the validation dataloader
        device: the device to run the model on
    returns:
        the average psnr of the validation dataset
    """
    total_psnr = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            batch_psnr = torch.mean(torch.stack([
                calculate_psnr_torch(gt, recon) 
                for gt, recon in zip(labels, outputs)
            ]))
            total_psnr += batch_psnr.item()
            count += 1

    return total_psnr / count

