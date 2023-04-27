import torch
from torchvision.transforms.functional import gaussian_blur

from .base import Corrupter


class GaussianBlur(Corrupter):
    def __init__(self, kernel_size=9, sigma=None):
        super().__init__()
        self.kernel_size = [kernel_size, kernel_size]
        self.sigma = sigma

    @torch.no_grad()
    def forward(self, image):
        return gaussian_blur(image, self.kernel_size, self.sigma)
