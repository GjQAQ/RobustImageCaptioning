from typing import *

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
        if kwargs.get('use_buffer', False):
            return self.decoder(None, None, None, mode='forward', use_buffer=True)

        fc, att = self.encode(image)
        # att_masks always is None because the size of att_feats is fixed
        if mode == 'forward':
            seq_per_img = sequence.size(0) // fc.size(0)
            fc = torch.repeat_interleave(fc, seq_per_img, 0)
            att = torch.repeat_interleave(att, seq_per_img, 0)
            # return log probabilities
            return self.decoder(fc, att, sequence, None, mode='forward', **kwargs)
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

    def encode(self, image: Union[torch.Tensor, List[torch.Tensor]]):
        # encoding cannot process in batch because images in MSCOCO have different sizes
        # while resnet can only encode images with same size in batch
        if isinstance(image, (list, tuple)):
            image_batch = image
        else:
            image_batch = (image,)

        fc_batch = []
        att_batch = []
        for image in image_batch:
            fc, att = self.encoder(image.squeeze().unsqueeze(0))
            fc_batch.append(fc)
            att_batch.append(att.reshape(1, -1, att.shape[-1]))
        return torch.cat(fc_batch), torch.cat(att_batch)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if 'decoder.buf_score_buffer' in state_dict:
            del state_dict['decoder.buf_score_buffer']
        return super().load_state_dict(state_dict, strict)

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

    @property
    def done_beams(self):
        return self.decoder.done_beams
