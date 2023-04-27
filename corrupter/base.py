import abc

from torch.nn import Module


class Corrupter(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, image):
        ...


class Preserve(Corrupter):
    def forward(self, image):
        return image
