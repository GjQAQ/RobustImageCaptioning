import torch

from .base import Corrupter


class Occlude(Corrupter):
    def __init__(self):
        super().__init__()
        # todo

    @torch.no_grad()
    def forward(self, image):
        c, h, w = image.shape
        pass
