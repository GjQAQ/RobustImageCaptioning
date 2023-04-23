import torch

from aoanet.models.AoAModel import AoAModel


class AttModelWrapper(AoAModel):
    def __init__(self, opt, temperature):
        super().__init__(opt)
        self.distilling_temperature = temperature

    def _sample(self, fc_feats, att_feats, att_masks=None, opt=None):
        if opt is None:
            opt = {}
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        unfinished = True
        it = 0
        if beam_size > 1:
            raise ValueError(f'beam_size in AttModelWrapper must be 1')

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seq_log_probs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # following cases are omitted:
            # if decoding_constraint and t > 0:
            # if remove_bad_endings and t > 0:
            # if block_trigrams and t >= 3:

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sample_log_probs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seq_log_probs[:, t] = sample_log_probs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seq_log_probs  # todo

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # only to apply temperature
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output) / self.distilling_temperature, dim=1)
        return logprobs, state
