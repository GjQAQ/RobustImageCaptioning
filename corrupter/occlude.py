from typing import *

import torch

from .base import Corrupter

__all__ = ['Occlude']


def _rand_index(limit):
    i = torch.randn(1) * limit / 8 + limit / 2
    i = torch.clamp(i, 0, limit - 1).int().item()
    i = (i + limit // 2) % limit
    return i


class Occlude(Corrupter):
    def __init__(
        self,
        size: Tuple[int, int] = None,
        ratio: Tuple[float, float] = (0.2, 0.2)
    ):
        super().__init__()
        self.size = size
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, image):
        h, w = image.shape[-2:]
        r0 = _rand_index(h)
        c0 = _rand_index(w)

        if self.size is not None:
            size = self.size
        else:
            size = tuple(map(int, (h * self.ratio[0], w * self.ratio[1])))
        half_h, half_w = size[0] // 2, size[1] // 2

        mask = torch.ones_like(image)
        mask[..., r0 - half_h:r0 + half_h, c0 - half_w:c0 + half_w] = 0
        return mask * image
