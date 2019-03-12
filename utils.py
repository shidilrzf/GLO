
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn.functional as F




def kernel_gauss(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")

    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyramid = []

    for level in range(max_levels):
        blurred = conv_gauss(current, kernel)
        diff = current - blurred
        pyramid.append(diff)
        current = F.avg_pool2d(blurred, 2)

    pyramid.append(current)
    return pyramid




class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, output, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != output.shape[1]:
            self._gauss_kernel = kernel_gauss(
                size=self.k_size, sigma=self.sigma,
                n_channels=output.shape[1], cuda=output.is_cuda
            )
        pyramid_output = laplacian_pyramid(output, self._gauss_kernel, self.max_levels)
        pyramid_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyramid_output, pyramid_target))


class IndexedDataset(Dataset):
    """
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)


def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum( z**2, axis=1))[:, np.newaxis], 1)


def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)

def save_checkpoint(state, filename):
    torch.save(state, filename)

