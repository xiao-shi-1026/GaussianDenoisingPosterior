import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import glob
import random
from .utils import data_augmentation
def normalize(data):
    return data / 255

def Im2Patch(img: np.array, patch_size: int, stride: int = 1) -> np.array:
    """
    Crop the image into small patches and return a set of patches. 
    All patches are squares with the side length patch_size.
    params:
        img: image in np.array with shape (channels, height, width)
        patch_size: the size of one patch
        stride: the stride of patches extraction. There's no overlap between patches if stride = patch_size
    return:
        Y: patches in shape (channels, patch_size, patch_size, # patches)
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
    k = 0 # k: the kth patch. Initialized here.
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    # Decide the output shape
    patch = img[:, 0:endw-patch_size+0+1:stride, 0:endh-patch_size+0+1:stride]
    # Calculate the output # patches
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, patch_size*patch_size,TotalPatNum], np.float32)
    for i in range(patch_size):
        for j in range(patch_size):
            patch = img[:, i:endw-patch_size + i + 1:stride,j:endh - patch_size + j + 1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, patch_size, patch_size, TotalPatNum])

def prepare_data(data_path: str, patch_size: int, stride : int = 1, aug_times: int = 1, mode: str = 'deblurring'):
    """
    Generate training data and validation data. 
    Training data will go through patching and data augmentation, and validation will not.
    Training set will be saved as train.h5 files. Validation data will be saved as val.h5
    params:
        data_path: original data location
        patch_size: the size of one patch
        stride: the stride of patches extraction. There's no overlap between patches if stride = patch_size
        aug_times: times of augments. default = 1
        mode: the subproblem at current stage
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

    # train
    print('process training data')
    # scaling list
    scales = [1, 0.9, 0.8, 0.7]
    # training location
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.transpose(Img, (2, 0, 1))
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, patch_size, stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data = data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'validation', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)


class ImageDataset(Dataset):
    def __init__(self, mode: str, train : bool = True):
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
        """ self.corruption = Blurkernel(blur_type = 'gaussian', 
                                        kernel_size = 3, 
                                        std = 10, 
                                        img_size = crop_size, 
                                        device = device)"""
        self.mode = mode
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[idx]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data), torch.Tensor(data)

if __name__ == "__main__":
    prepare_data(r"C:\Users\sx119\Desktop\gdp_data", 128, 100)