import torch
from torch.nn import Module
from torch.nn.functional import kl_div

from .e2e import FixedFeatureCaptionModel
from utils.check import *
from aoanet.misc.utils import LabelSmoothing


class DistillationContainer(Module):
    def __init__(
        self,
        teacher: FixedFeatureCaptionModel,
        student: FixedFeatureCaptionModel,
        corrupter,
        temperature=20.0,
        hard_weight=0.1,
        smoothing=0.0
    ):
        super().__init__()
        self.__temperature = temperature
        self.__hard_weight = hard_weight

        self.teacher = teacher
        self.student = student
        self.corrupter = corrupter
        self.hard_loss = LabelSmoothing(smoothing=smoothing)

        self.teacher.eval()
        self.teacher.temperature = temperature
        self.student.train()

    def forward(self, image, sequence, mask):
        with torch.no_grad():
            reference = self.teacher(image, sequence)[2]

        corrupted = self.corrupter(image)
        self.student.temperature = self.temperature
        soft_prob = self.student(corrupted, sequence)
        self.student.temperature = 1
        hard_prob = self.student(corrupted, sequence)

        soft_loss = kl_div(soft_prob, reference, reduction='none', log_target=True)  # todo
        hard_loss = self.hard_loss(hard_prob, labels[:, 1:], mask[:, 1:])

        return soft_loss / self.temperature ** 2 + self.__hard_weight * hard_loss

    def train(self, mode=True):
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
