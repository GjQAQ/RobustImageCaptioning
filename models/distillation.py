import torch
from torch.nn import Module
from torch.nn.functional import kl_div

from .e2e import FixedFeatureCaptionModel
import utils
from utils.check import *
from corrupter import Corrupter
from aoanet.misc.utils import LabelSmoothing


class DistillationContainer:
    def __init__(
        self,
        teacher: FixedFeatureCaptionModel,
        student: FixedFeatureCaptionModel,
        corrupter: Corrupter,
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
            # reference = self.teacher(image, sequence, 'distill')
            reference = self.teacher(image, sequence)

        if isinstance(image, torch.Tensor):
            corrupted = self.corrupter(image)
        else:
            corrupted = list(map(self.corrupter, image))
        self.student.temperature = self.temperature
        soft_prob = self.student(corrupted, sequence)
        self.student.temperature = 1
        hard_prob = self.student(corrupted, sequence, use_buffer=True)

        mask = mask[:, 1:]
        soft_loss = kl_div(soft_prob, reference, reduction='none', log_target=True)
        soft_loss = (soft_loss.sum(2) * mask).sum() / mask.sum()
        hard_loss = self.hard_loss(hard_prob, sequence[:, 1:], mask)

        final = soft_loss / self.temperature ** 2 + self.__hard_weight * hard_loss
        return final, soft_loss, hard_loss

    @property
    def temperature(self):
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        _check_temperature(value)
        self.__temperature = value
        self.teacher.temperature = value
