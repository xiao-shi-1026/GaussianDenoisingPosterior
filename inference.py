import torch
from torchvision import transforms
from PIL import Image
from model.DnCNN import DnCNN
from model.UNet import UNet
from pathlib import Path
import matplotlib.pyplot as plt
from data.blurring import corrupt
from collections import OrderedDict
import torch.nn.functional as F
from data.utils import addnoise
from data.inpaint import build_inpaint_freeform
import data.PSNR as PSNR
def load_model(model_class, model_path, device):
    model = model_class()
    checkpoint = torch.load(model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
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

def inference_pipeline(model, image_path, inpaint_fn, device="cuda"):

    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    corrupted_tensor, _ = inpaint_fn(input_tensor)
    corrupted_tensor = addnoise(corrupted_tensor, [5, 5], device)

    denoised_tensor = run_inference(model, corrupted_tensor, device)
    denoised_image = postprocess_output(denoised_tensor)

    psnr_value = PSNR.calculate_psnr_torch(input_tensor, denoised_tensor)
    return original_image, corrupted_tensor, denoised_image, psnr_value

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

def plot_real(original_image, denoised_image):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image") 

    axes[1].imshow(denoised_image)
    axes[1].set_title("Denoised Image")

    for ax in axes:
        ax.axis("off")
    plt.show()

def run_on_folder(model_path, image_folder, device="cuda"):
    class Opt:
        image_size = 256
        device = device
    opt = Opt()
    mask_type = 'freeform1020' 
    inpaint_fn = build_inpaint_freeform(opt, mask_type)

    model = load_model(UNet, model_path, device)

    image_paths = list(Path(image_folder).glob("*.[pj][pn]g"))  # 支持 jpg 和 png
    total_psnr = 0
    count = 0

    for image_path in image_paths:
        print(f"Inferencing {image_path.name}")
        try:
            original_image, corrupted_tensor, denoised_image, psnr_value = inference_pipeline(model, image_path, inpaint_fn, device)
            total_psnr += psnr_value.item()
            count += 1
            # plot_images(original_image, corrupted_tensor, denoised_image)  # 可选，注释掉可加快速度
        except Exception as e:
            print(f"Failed on {image_path.name}: {e}")

    if count > 0:
        avg_psnr = total_psnr / count
        print(f"\nAverage PSNR over {count} images: {avg_psnr:.2f}")
    else:
        print("No valid images processed.")


if __name__ == "__main__":

    model_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\outputs\inpainting_unet_2.pth"
    image_folder = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\data\ffhq256\validation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_on_folder(model_path, image_folder, device) 
