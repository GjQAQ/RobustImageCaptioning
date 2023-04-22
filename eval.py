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
import aoanet.opts as opts
from aoanet.models.AoAModel import AoAModel
from dataloader import *
from aoanet.dataloaderraw import *
import aoanet.eval_utils as eval_utils
import aoanet.misc.utils as misc_utils
from models import ResNet101Encoder
from utils.loader import DataloaderWrapper


def extract_state_dict(ext_dict, key):
    res = {}
    for k, v in ext_dict.items():
        if k.startswith(f'{key}.'):
            res[k] = v
    return res


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
    help='path to model to evaluate')
parser.add_argument('--infos_path', type=str, default='',
    help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()

dataset_root = '/root/autodl-tmp/datasets/MSCOCO'
device = 'cuda'

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = misc_utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
opt.vocab = vocab
encoder = ResNet101Encoder(7)
decoder = AoAModel(opt)
del opt.vocab
model = nn.ModuleDict({
    'encoder': encoder,
    'decoder': decoder
})
model.load_state_dict(torch.load(opt.model))
encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()
crit = misc_utils.LanguageModelCriterion()

loader = DataloaderWrapper(encoder, dataset_root, opt, device=device)
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(decoder, crit, loader, vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
