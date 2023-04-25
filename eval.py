from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import os
import argparse
from six.moves import cPickle

import numpy as np
import torch
import torch.nn as nn

import adapter
import utils
import models
from aoanet.dataloaderraw import *
import aoanet.eval_utils as eval_utils
import aoanet.misc.utils as misc_utils

__replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
__ignore = ['start_from']


def extract_state_dict(ext_dict, key):
    res = {}
    for k, v in ext_dict.items():
        if k.startswith(f'{key}.'):
            res[k] = v
    return res


def prepare_opt(opt):
    # Load infos
    with open(opt.infos_path, 'rb') as f:
        infos = misc_utils.pickle_load(f)
    for k in vars(infos['opt']).keys():
        if k in __replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in __ignore:
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})
    return opt, info


if __name__ == '__main__':
    parser = utils.set_eval_parser()
    opt = parser.parse_args()

    dataset_root = opt.dataset_root
    device = opt.evaluation_device

    opt, info = prepare_opt(opt)
    vocab = infos['vocab']  # ix -> word mapping

    # Setup the model
    opt.vocab = vocab
    encoder = models.ResNet101Encoder(7)
    decoder = models.AoAModelWrapper(opt, 1.0)
    del opt.vocab
    model = models.FixedFeatureCaptionModel(encoder, decoder, {})
    model.load_state_dict(torch.load(opt.model))
    crit = misc_utils.LanguageModelCriterion()

    loader = utils.DataloaderWrapper(encoder, dataset_root, opt, device=device)
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.ix_to_word = infos['vocab']

    opt.datset = opt.input_json
    evaluation_args = vars(opt)
    model.evaluation_args.update(evaluation_args)
    model.to(device)
    model.eval()
    loss, split_predictions, lang_stats = eval_utils.eval_split(decoder, crit, loader, evaluation_args)

    print('loss: ', loss)
    if lang_stats:
        print(lang_stats)

    if opt.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open('vis/vis.json', 'w'))
