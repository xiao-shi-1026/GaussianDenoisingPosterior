# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from diffusion_palette_eval.
#
# Source:
# https://bit.ly/eval-pix2pix
#
# ---------------------------------------------------------------

import io
import math
from PIL import Image, ImageDraw

import os

import numpy as np
import torch

from pathlib import Path
import gdown


FREEFORM_URL = "https://drive.google.com/file/d/1-5YRGsekjiRKQWqo0BV5RVQu0bagc12w/view?usp=share_link"

# code adoptted from
# https://bit.ly/eval- pix2pix
def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask

# code adoptted from
# https://bit.ly/eval-pix2pix
def load_masks(filename):
    # filename = "imagenet_freeform_masks.npz"
    shape = [10000, 256, 256]

    # shape = [10950, 256, 256] # Uncomment this for places2.

    # Load the npz file.
    with open(filename, 'rb') as f:
        data = f.read()

    data = dict(np.load(io.BytesIO(data)))
    # print("Categories of masks:")
    # for key in data:
    #     print(key)

    # Unpack and reshape the masks.
    for key in data:
        data[key] = np.unpackbits(data[key], axis=None)[:np.prod(shape)].reshape(shape).astype(np.uint8)

    # data[key] contains [10000, 256, 256] array i.e. 10000 256x256 masks.
    return data

def load_freeform_masks(op_type):
    """
    Download the original imagenet_freeform_masks.npz.
    Extract the given name masks from the original data
    The Extracted data is named as imagenet_freeform{key}_masks.npz
    
    """
    data_dir = Path("data")

    mask_fn = data_dir / f"imagenet_{op_type}_masks.npz"
    if not mask_fn.exists():
        # download orignal npz from palette google drive
        orig_mask_fn = str(data_dir / "imagenet_freeform_masks.npz")
        if not os.path.exists(orig_mask_fn):
            gdown.download(url=FREEFORM_URL, output=orig_mask_fn, quiet=False, fuzzy=True)
        masks = load_masks(orig_mask_fn)

        # store freeform of current ratio for faster loading in future
        key = {
            "freeform1020": "10-20% freeform",
            "freeform2030": "20-30% freeform",
            "freeform3040": "30-40% freeform",
        }.get(op_type)
        np.savez(mask_fn, mask=masks[key])

    # [10000, 256, 256] --> [10000, 1, 256, 256]
    return np.load(mask_fn)["mask"][:,None]

def get_center_mask(image_size):
    h, w = image_size
    mask = bbox2mask(image_size, (h//4, w//4, h//2, w//2))
    return torch.from_numpy(mask).permute(2,0,1)

def build_inpaint_center(opt, mask_type):
    assert mask_type == "center"

    center_mask = get_center_mask([opt.image_size, opt.image_size])[None,...] # [1,1,256,256]
    center_mask = center_mask.to(opt.device)

    def inpaint_center(img):
        # img: [-1,1]
        mask = center_mask
        # img[mask==0] = img[mask==0], img[mask==1] = 1 (white)
        return img * (1. - mask) + mask, mask

    return inpaint_center

def build_inpaint_freeform(opt, mask_type):
    assert "freeform" in mask_type

    freeform_masks = load_freeform_masks(mask_type) # [10000, 1, 256, 256]
    n_freeform_masks = freeform_masks.shape[0]
    freeform_masks = torch.from_numpy(freeform_masks).to(opt.device)

    def inpaint_freeform(img):
        # img: [-1,1]
        index = np.random.randint(n_freeform_masks, size=img.shape[0])
        mask = freeform_masks[index]
        # img[mask==0] = img[mask==0], img[mask==1] = 1 (white)
        return img * (1. - mask) + mask, mask

    return inpaint_freeform

if __name__ == '__main__':
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    class Opt:
        image_size = 256
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    def show_image(tensor, title="Image"):
        tensor = tensor.squeeze(0).detach().cpu()  # [3,H,W]
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        img = T.ToPILImage()(tensor)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()

    opt = Opt()
    path = 'data/sample_data'

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),  # [0,1]
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),  # 归一化到 [-1,1]
    ])
    mask_type = 'freeform3040' 
    inpaint_fn = build_inpaint_freeform(opt, mask_type)
    for filename in os.listdir(path):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(path, filename)
        image = Image.open(img_path).convert("RGB")
        img = transform(image).unsqueeze(0).to('cuda')
        corrupted_img, mask = inpaint_fn(img)
        show_image(img, "Original Image")
        show_image(corrupted_img, "Corrupted Image")

def generate_mask(batch_size: int, image_shape: tuple, drop_ratio: float = 0.15, RGB: bool = True) -> torch.Tensor:
    """
    Generate a batch of binary masks for images with a given shape.

    Args:
        batch_size: int, number of masks to generate
        image_shape: tuple, shape of the image (H, W)
        drop_ratio: float, percentage of pixels to be masked
        RGB: bool, if True, output shape is (B, 3, H, W), otherwise (B, 1, H, W)

    Returns:
        mask_batch: torch.Tensor, binary mask (1 = masked, 0 = keep)
    """
    H, W = image_shape
    total_pixels = H * W
    num_masked = int(drop_ratio * total_pixels)

    num_channels = 3 if RGB else 1
    mask_batch = torch.zeros((batch_size, num_channels, H, W), dtype=torch.uint8)

    for b in range(batch_size):
        flat_mask = torch.zeros(total_pixels, dtype=torch.uint8)
        mask_indices = torch.randperm(total_pixels)[:num_masked]
        flat_mask[mask_indices] = 1
        base_mask = flat_mask.view(1, H, W)  # shape: [1, H, W]

        # repeat across channels
        mask_batch[b] = base_mask.expand(num_channels, H, W)

    return mask_batch

def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a binary mask to a single image. Masked regions will be white (1.0).

    Args:
        image: torch.Tensor, shape (3, H, W), input image in [0,1]
        mask: torch.Tensor, shape (1, H, W) or (3, H, W), binary mask (1 = masked)

    Returns:
        masked_img: torch.Tensor, shape (3, H, W)
    """
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.expand(3, -1, -1)  # [3, H, W]

    # Apply mask: 1 → white, 0 → original pixel
    masked_img = image * (1.0 - mask) + mask

    return masked_img