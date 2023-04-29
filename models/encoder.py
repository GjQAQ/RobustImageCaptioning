import abc
import functools

import torch
from torch.nn import Module, Sequential
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.modules.module import T

import aoanet.misc.resnet as resnets


class Encoder(Module, metaclass=abc.ABCMeta):
    def __init__(self, attention_size, pretrained=True):
        super().__init__()
        self.attention_size = attention_size  # A

        base = self.base_constructor[0](pretrained)
        self.layers = Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4
        )

    def forward(self, x):
        x = self.layers(x)
        fc = x.mean(3).mean(2).squeeze()[None, ...]  # add batch dimension
        att = adaptive_avg_pool2d(x, self.attention_size).permute(0, 2, 3, 1)  # B x A x A x C

        return fc, att

    def train(self: T, mode: bool = True) -> T:
        if mode:
            return super().train(mode)


def define(name, f):
    # if binding f rather than (f,), 'self' will be passed as the first argument
    # when calling self.base_constructor in ResNetBaseEncoder
    return type(name, (Encoder,), {'base_constructor': (f,)})


ResNet18Encoder = define('ResNet18Encoder', resnets.resnet18)
ResNet34Encoder = define('ResNet34Encoder', resnets.resnet34)
ResNet50Encoder = define('ResNet50Encoder', resnets.resnet50)
ResNet101Encoder = define('ResNet101Encoder', resnets.resnet101)
ResNet152Encoder = define('ResNet152Encoder', resnets.resnet152)
