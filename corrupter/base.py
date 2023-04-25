import abc

from torch.nn import Module


class Corrupter(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, image):
        ...
