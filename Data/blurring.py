import torch
from torch import nn
import scipy
import numpy as np
from abc import ABC, abstractmethod

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

class Blurkernel(nn.Module):
    def mat_by_img(self, M, v):

        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, img_size = 64,  device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3).to(self.device)
        )
        self.weights_init()
        self.construct_1d_convolution_matrix_from_2d_kernel(img_dim=img_size)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k.to(self.device)
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k.to(self.device)
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k.to(self.device)

    def construct_1d_convolution_matrix_from_2d_kernel(self, img_dim, direction='row', ZERO = 3e-2):
        assert self.kernel_size % 2 == 1, "Kernel size must be odd."

        if direction == 'row':
            kernel_1d = self.k[self.kernel_size // 2, :]  # Center row
        elif direction == 'col':
            kernel_1d = self.k[:, self.kernel_size // 2]  # Center column
        else:
            raise ValueError("Direction must be 'row' or 'col'.")

        kernel_1d = kernel_1d / kernel_1d.sum()

        pad_size = self.kernel_size // 2
        padded_kernel = torch.zeros(img_dim + 2 * pad_size)
        padded_kernel[:self.kernel_size] = kernel_1d

        H = torch.zeros((img_dim, img_dim), device = self.device)
        for i in range(img_dim):
            shifted_kernel = torch.roll(padded_kernel, shifts=i, dims=0)
            cropped_kernel = shifted_kernel[pad_size:pad_size + img_dim]
            H[i, :] = cropped_kernel

        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.singulars_small[self.singulars_small < ZERO] = 0
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1),
                                       self.singulars_small.reshape(1, img_dim)).reshape(img_dim ** 2)
        self._singulars, self._perm = self._singulars.sort(descending=True)
        self.H = H

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def compute_ybar( self, y):
        U_small = self.U_small  # U matrix (1D convolution)
        singulars_small = self.singulars_small  # Singular values
        Sigma_plus = torch.zeros_like(singulars_small)
        Sigma_plus[singulars_small > 0] = 1 / singulars_small[singulars_small > 0]
        UT_y = torch.matmul(U_small.T, y)
        ybar = Sigma_plus * UT_y
        return ybar
    
def corrupt(im: torch.Tensor, device: torch.device):
    with torch.no_grad():
        corruption = Blurkernel(blur_type = 'gaussian', 
                                    kernel_size = 15, 
                                    std = 15, 
                                    img_size = im.shape[0], 
                                    device = device)
        im = im.to(device)
        corruption = corruption.forward(im)
        torch.cuda.empty_cache()
        return corruption
