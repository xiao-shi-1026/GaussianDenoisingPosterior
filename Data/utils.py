import torch
import numpy as np

def addnoise(img_train: torch.Tensor, noiseL_B: list, device: torch.device) -> torch.Tensor:
    """
    Add random level of gaussian noise.
    params:
        img_train: image to add noise
        noise_L_B: the range of noise levels
    returns:
        image after adding noise
    """
    noise = torch.zeros(img_train.size())
    stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
    for n in range(noise.size()[0]):
        sizeN = noise[0,:,:,:].size()
        noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
    return (img_train + noise.to(device)).clamp(0, 1) # limit to [0,1]

def data_augmentation(image, mode):
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
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))