import torch
from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from data.blurring import corrupt

def calculate_psnr_torch(img1: torch.Tensor, img2: torch.Tensor, max_pixel: float = 1.0) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((max_pixel ** 2) / mse)

def evaluate_psnr_torch(val_loader: DataLoader, device: torch.device) -> float:
    total_psnr = 0.0
    count = 0

    for original, reconstructed in val_loader:
        original, reconstructed = original.to(device), reconstructed.to(device)

        # change this
        reconstructed = corrupt(reconstructed)

        batch_psnr = torch.mean(torch.stack([calculate_psnr_torch(orig, recon) for orig, recon in zip(original, reconstructed)]))
        total_psnr += batch_psnr.item()
        count += 1

    return total_psnr / count

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    test_dataset = ImageDataset('/Users/sx/Desktop/GaussianDenoisingPosterior/data/sample_data')
    test_loader = DataLoader(dataset=test_dataset, batch_size=3, shuffle=False, num_workers=1)
    evaluate_psnr_torch(test_loader, "cuda" if torch.cuda.is_available() else "cpu")

