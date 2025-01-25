import torch
from torchvision import transforms
from PIL import Image
from model.DnCNN import DnCNN
from pathlib import Path
import matplotlib.pyplot as plt
from data.blurry import corrupt

def load_model(model_class, model_path, device):
    model = model_class(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='BR')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform = None):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image_tensor = transform(image).unsqueeze(0)
    else:
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    return image_tensor, image

def run_inference(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output

def postprocess_output(output_tensor):
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = (output_image * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(output_image)

def inference_pipeline(model_path, image_path, device="cuda"):
    model = load_model(DnCNN, model_path, device)

    input_tensor, original_image = preprocess_image(image_path)

    corrupted_tensor = corrupt(input_tensor, device)

    denoised_tensor = run_inference(model, corrupted_tensor, device)

    denoised_image = postprocess_output(denoised_tensor)

    plot_images(original_image, corrupted_tensor, denoised_image)

def plot_images(original_image, corrupted_tensor, denoised_image):

    corrupted_image = corrupted_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    corrupted_image = (corrupted_image * 255).clip(0, 255).astype("uint8")
    corrupted_image = Image.fromarray(corrupted_image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[1].imshow(corrupted_image)
    axes[1].set_title("Corrupted Image")
    axes[2].imshow(denoised_image)
    axes[2].set_title("Denoised Image")

    for ax in axes:
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    model_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\outputs\deblurring\deblurring.pth"
    image_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\data\test_image.png.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_pipeline(model_path, image_path, device)
