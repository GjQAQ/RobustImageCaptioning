import torch
import torch.nn

from .encoder import Encoder
from .decoder import AoAModelWrapper
from utils.check import *


class FixedFeatureCaptionModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: AoAModelWrapper, eval_args: dict):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.__evaluation_args = eval_args
        self.__temperature = 1.0

    def forward(self, image, sequence, mode='forward', **kwargs):
        fc, att = self.encoder(image)
        # att_masks always is None because the size of att_feats is fixed
        if mode == 'forward':
            # return log probabilities
            return self.decoder(fc, att, sequence, None, mode='forward', use_buffer=kwargs['use_buffer'])
        elif mode == 'sample':
            # return sequence and log probabilities in sequence
            return self.decoder(fc, att, None, opt=self.evaluation_args, mode='sample')
        elif mode == 'distill':
            if self.evaluation_args['beam_size'] != 1:
                raise ValueError(f'beam_size must be 1 during distillation')
            # return log probability distribution
            return self.decoder(fc, att, None, opt=self.evaluation_args, mode='distill')
        else:
            raise ValueError(f'Unknown mode: {mode}')

    @property
    def temperature(self):
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        _check_temperature(value)

        self.__temperature = value
        self.decoder.distilling_temperature = value  # for AttModel._forward
        self.__evaluation_args['temperature'] = value  # for AttModel._sample

    @property
    def evaluation_args(self):
        return self.__evaluation_args
