import torch
from torch.nn import Module
from torch.nn.modules.module import T

from .e2e import FixedFeatureCaptionModel
from utils.check import *


class DistillationContainer(Module):
    def __init__(
        self,
        teacher: FixedFeatureCaptionModel,
        student: FixedFeatureCaptionModel,
        corrupter,
        temperature=20.0,
        hard_weight=0.1
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.corrupter = corrupter
        self.__temperature = temperature
        self.__hard_weight = hard_weight

        self.teacher.eval()
        self.teacher.temperature = temperature
        self.student.train()

    def forward(self, image, sequence):
        reference = self.teacher(image, sequence)[1]

        corrupted = self.corrupter(image)
        self.student.temperature = self.temperature
        soft_prob = self.student(image, sequence)  # todo:multiply soft by T^2
        self.student.temperature = 1
        hard_prob = self.student(image, sequence)

    def train(self: T, mode: bool = True) -> T:
        if not mode:
            raise RuntimeError('Distillation container can only be used to train')
        return super().train(mode)

    @property
    def temperature(self):
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        _check_positive(value)
        self.__temperature = value
        self.teacher.temperature = value
