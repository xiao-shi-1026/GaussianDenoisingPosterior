"""
References:
    @article{zhang2017beyond,
    title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
    journal={IEEE Transactions on Image Processing},
    volume={26},
    number={7},
    pages={3142--3155},
    year={2017},
    publisher={IEEE}
    }
"""
import os
import cv2
import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset


def normalize(data):
    return data / 255

def prepare_data(data_path: str, aug_times: int = 1):
    """
    Generate training and validation datasets in HDF5 format with progress bar.
    Each sample has shape (3, H, W), with data augmentation applied.
    
    Args:
        data_path (str): Path to dataset.
        aug_times (int): Number of augmentations per image.
    """
    # Prepare train data
    print('Processing training data...')
    train_files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    train_files.sort()

    with h5py.File('train.h5', 'w') as h5f:
        index = 0
        for file in tqdm(train_files, desc="Processing Train Images"):  # 加入 tqdm 进度条
            img = cv2.imread(file)  # Read image (H, W, 3) in BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = np.transpose(img, (2, 0, 1))  # Change to (3, H, W)
            img = np.float32(normalize(img))  # Normalize to [0,1]

            # Store original image
            h5f.create_dataset(str(index), data=img)
            index += 1

            # Apply augmentations
            for i in range(aug_times):
                mode = np.random.randint(0, 8)  # Randomly select augmentation mode
                aug_img = data_augmentation(img, mode)
                h5f.create_dataset(f"{index}_aug_{i}", data=aug_img)
                index += 1

    print(f'Training set saved with {index} samples (including augmentations).')

    # Prepare validation data (no augmentation)
    print('Processing validation data...')
    val_files = glob.glob(os.path.join(data_path, 'validation', '*.png'))
    val_files.sort()

    with h5py.File('val.h5', 'w') as h5f:
        for i, file in enumerate(tqdm(val_files, desc="Processing Validation Images")):
            img = cv2.imread(file)  # Read image (H, W, 3) in BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = np.transpose(img, (2, 0, 1))  # Change to (3, H, W)
            img = np.float32(normalize(img))  # Normalize to [0,1]

            # Store validation image (no augmentation)
            h5f.create_dataset(str(i), data=img)

    print(f'Validation set saved with {len(val_files)} samples.')


class ImageDataset(Dataset):
    def __init__(self, h5_path: str, train: bool = True, transform=None):
        """
        params:
            h5_path: HDF5 path ('train.h5' or 'val.h5')
            train: if training, shuffle
            transform: preprocess
        """
        self.h5_path = h5_path
        self.train = train
        self.transform = transform

        with h5py.File(self.h5_path, 'r') as h5f:
            self.keys = list(h5f.keys())
        # shuffle
        if self.train:
            random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        return (3, H, W) tensor
        """
        with h5py.File(self.h5_path, 'r') as h5f:
            key = self.keys[idx]
            data = np.array(h5f[key]) # hdf5 with shape = (3, H, W)

        # Transform as PyTorch Tensor
        data = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)

        return data, data

if __name__ == "__main__":
    from utils import data_augmentation
    from tqdm import tqdm
    import glob
    prepare_data(r"C:\Users\sx119\Desktop\ffhq256")