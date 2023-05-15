import torch
import torch.nn
from torch.nn.functional import log_softmax

from utils.check import *
from aoanet.models.AoAModel import AoAModel


class AoAModelWrapper(AoAModel):
    def __init__(self, opt, temperature=1.0):
        super().__init__(opt)
        self.__distilling_temperature = temperature

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, use_buffer=False):
        """
        Enable buffering scores before softmax to facilitate student's forward.
        """
        if use_buffer:
            return log_softmax(self.score_buffer / self.distilling_temperature, dim=2)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)
        self.register_buffer('buf_score_buffer', torch.zeros_like(outputs), persistent=False)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            score, state = self.__get_score(
                it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state
            )
            self.score_buffer[:, i] = score
            outputs[:, i] = log_softmax(score / self.distilling_temperature, dim=1)

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        """Enable temeprature"""
        it = it.to(self.device)
        score, state = self.__get_score(it, fc_feats, att_feats, p_att_feats, att_masks, state)
        return log_softmax(score / self.distilling_temperature, dim=1), state

    def __get_score(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        return self.logit(output), state

    @property
    def distilling_temperature(self):
        return self.__distilling_temperature

    @distilling_temperature.setter
    def distilling_temperature(self, value):
        _check_temperature(value)
        self.__distilling_temperature = value

    @property
    def score_buffer(self):
        return self.buf_score_buffer

    @property
    def device(self):
        sub: torch.nn.Linear = self.ctx2att
        return sub.weight.device
