import torch
from torch.nn import Module

from aoanet.models.AttModel import AttModel


class DistillationContainer(Module):
    def __init__(self, teacher, student, temperature=20.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature

    def forward(self, images):
        pass  # todo


class Teacher(Module):
    def __init__(self, base, temperature=20.0):
        super().__init__()
        self.base = base
        self.base.temperature = temperature

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.base.forward(*args, **kwargs)


class Student(Module):
    def __init__(self, base, temperature=20.0):
        super().__init__()
        self.base = base
        self.temperature = temperature

    def forward(self, *args, **kwargs):
        self.base.temperature = 1
        sincere = self.base(*args, **kwargs)
