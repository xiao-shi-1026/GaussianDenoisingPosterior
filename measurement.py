
from data.inpaint import build_inpaint_freeform
import torch
import inference as inf
from model.UNet import UNet
from data.utils import addnoise
import data.PSNR as PSNR
def generate_measurement(img: torch.Tensor, mask_type: str, number: int, opt) -> torch.Tensor:
    """
    Generate measurement from image using inpainting method.
    
    Parameters:
        img (torch.Tensor): Input image tensor of shape (B, C, H, W).
        mask_type (str): Type of mask to use for inpainting.
        number (int): Number of measurements to generate.
        opt: Options containing image size and device information.

    Returns:
        torch.Tensor: Measurement tensor of shape (B, C, H, W).
    """
    # Ensure the input image is in the correct shape
    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Check if the input image has the correct number of channels
    if img.size(1) != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    # Build inpainting function based on mask type
    inpaint_fn = build_inpaint_freeform(opt, mask_type)

    # Generate measurements using inpainting function
    measurements = []
    for _ in range(number):
        measurement, _ = inpaint_fn(img)
        measurements.append(measurement)

    return torch.cat(measurements, dim=0)  # Concatenate along batch dimension
    
def inference_pipeline(model, measurement, device):
    """
    Inference pipeline for denoising images using the trained model.
    
    Parameters:
        model: Trained model for denoising.
        measurement (torch.Tensor): Input measurement tensor of shape (B, C, H, W).
        device: Device to run the model on.

    Returns:
        tuple: Original image, corrupted tensor, denoised image, and PSNR value.
    """
    # Move measurement to the specified device
    measurement = measurement.to(device)

    # Run inference using the model
    denoised_tensor = model(measurement)

    # Calculate PSNR value
    psnr_value = PSNR.calculate_psnr_torch(measurement, denoised_tensor)

    return denoised_tensor, psnr_value

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class Opt:
        image_size = 256
        device = device

    opt = Opt()
    mask_type = 'freeform1020' 
    number = 10
    img = torch.randn(1, 3, 256, 256).to(device)  # Example input image tensor
    img = img.to(device)
    img = addnoise(img, [5, 5], device)
    measurement = generate_measurement(img, mask_type, number, opt)
    print(measurement.shape)  # Should be (10, 3, 256, 256)
    measurement = measurement.to(device)
    # iterate through the measurement tensor and apply the inference pipeline
    model_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\outputs\inpainting_unet_2.pth"
    model = inf.load_model(UNet, model_path, device)
    avg_psnr = 0.0
    count = 0
    for i in range(measurement.shape[0]):
        measurement_i = measurement[i:i+1]
        denoised_image, psnr_value = inference_pipeline(model, measurement_i, device)
        print(f"PSNR for measurement {i}: {psnr_value.item()}")
        avg_psnr += psnr_value.item()
        count += 1
    print(f"Average PSNR: {avg_psnr / count:.2f}")
    # plot_images(original_image, corrupted_tensor, denoised_image)  # Optional, comment out to speed up

