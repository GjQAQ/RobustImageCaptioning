import torch
import torch.nn

from .encoder import Encoder
from .decoder import AttModelWrapper
from utils.check import *


class FixedFeatureCaptionModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: AttModelWrapper, eval_args: dict):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.__evaluation_args = eval_args
        self.__temperature = 1.0

    def forward(self, image, sequence):
        fc, att = self.encoder(image)
        # att_masks always is None because the size of att_feats is fixed
        if self.training:
            return self.decoder(fc, att, sequence, None)  # log probabilities

        if self.evaluation_args['beam_size'] != 1:
            raise ValueError()
        # sequence, log probabilities of output and log probability distribution
        return self.decoder(fc, att, None, opt=self.evaluation_args, mode='sample')

    @property
    def temperature(self):
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        _check_positive(value)

        self.__temperature = value
        self.decoder.distilling_temperature = value  # for AttModel._forward
        self.__evaluation_args['temperature'] = value  # for AttModel._sample

    @property
    def evaluation_args(self):
        return self.__evaluation_args
