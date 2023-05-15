import torch

from .blur import Corrupter


class Noise(Corrupter):
    def __init__(self, intensity: float = 0.1):
        super().__init__()
        self.__intensity = intensity

    def forward(self, image):
        noise = torch.randn_like(image)
        noise *= self.__intensity * image.max()

        image += noise
        image = torch.clamp(image, 0, None)
        return image
