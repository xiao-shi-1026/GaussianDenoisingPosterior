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

def evaluate_psnr_torch(val_loader: DataLoader, device: torch.device) -> float:
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

    for original, reconstructed in val_loader:
        original, reconstructed = original.to(device), reconstructed.to(device)

        # change this
        reconstructed = corrupt(reconstructed, device)

        batch_psnr = torch.mean(torch.stack([calculate_psnr_torch(orig, recon) for orig, recon in zip(original, reconstructed)]))
        total_psnr += batch_psnr.item()
        count += 1

    return total_psnr / count

if __name__ == '__main__':
    # test
    test_dataset = ImageDataset(r'C:\Users\sx119\Desktop\GaussianDenoisingPosterior\data\sample_data')
    test_loader = DataLoader(dataset=test_dataset, batch_size=3, shuffle=False, num_workers=1)
    result = evaluate_psnr_torch(test_loader, "cuda" if torch.cuda.is_available() else "cpu")
    print(result)

