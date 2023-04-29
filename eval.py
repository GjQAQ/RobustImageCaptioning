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
import corrupter
from aoanet.dataloaderraw import *
import aoanet.eval_utils as eval_utils
import aoanet.misc.utils as misc_utils

__replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
__ignore = ['start_from']


def prepare_opt(opt):
    # Load infos
    with open(opt.infos_path, 'rb') as f:
        infos = misc_utils.pickle_load(f)
    for k in vars(infos['opt']).keys():
        if k in __replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in __ignore:
            if k not in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})
    return opt, infos


def dump(sents, data, eval_kwargs, predictions, verbose):
    for k, sent in enumerate(sents):
        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
        if eval_kwargs.get('dump_path', 0) == 1:
            entry['file_name'] = data['infos'][k]['file_path']
        predictions.append(entry)
        if eval_kwargs.get('dump_images', 0) == 1:
            # dump the raw image to vis/ folder
            cmd = 'cp "' + os.path.join(
                eval_kwargs['image_root'],
                data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg'  # bit gross
            print(cmd)
            os.system(cmd)

        if verbose:
            print('image %s: %s' % (entry['image_id'], entry['caption']))


def print_beam(loader, model):
    for i in range(loader.batch_size):
        print('\n'.join([
            misc_utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0]
            for _ in model.done_beams[i]
        ]))
        print('--' * 10)


# a variant of aoanet.eval_utils.eval_split
@torch.no_grad()
def evaluate(model, crit, loader, eval_kwargs):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    crrupter = corrupter.get_instance(eval_kwargs['corrupter'])  # blur only
    # Use this nasty way to make other code clean since it's a global configuration
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split, image_only=True)
        data['images'] = list(map(crrupter, data['images']))
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            with torch.no_grad():
                loss = crit(
                    model(data['images'], data['labels']),
                    data['labels'][:, 1:],
                    data['masks'][:, 1:]
                ).item()
                loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(data['images'], None, opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            print_beam(loader, model)

        dump(misc_utils.decode_sequence(loader.get_vocab(), seq), data, eval_kwargs, predictions, verbose)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped'] or num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = eval_utils.language_eval(dataset, predictions, eval_kwargs['id'], split)
    return loss_sum / loss_evals, predictions, lang_stats


if __name__ == '__main__':
    parser = utils.set_eval_parser()
    opt = parser.parse_args()

    dataset_root = opt.dataset_root
    device = opt.evaluation_device

    opt, infos = prepare_opt(opt)
    vocab = infos['vocab']  # ix -> word mapping

    # Setup the model
    opt.vocab = vocab
    encoder = models.ResNet101Encoder(7)
    decoder = models.AoAModelWrapper(opt, 1.0)
    model = models.FixedFeatureCaptionModel(encoder, decoder, {})
    model.load_state_dict(torch.load(opt.model))
    crit = misc_utils.LanguageModelCriterion()
    del opt.vocab

    loader = utils.DataloaderWrapper(encoder, dataset_root, opt, device=device)
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.ix_to_word = infos['vocab']

    opt.datset = opt.input_json
    evaluation_args = vars(opt)
    model.evaluation_args.update(evaluation_args)
    model.to(device)
    model.eval()
    encoder.train()
    loss, split_predictions, lang_stats = evaluate(model, crit, loader, evaluation_args)
    # loss, split_predictions, lang_stats = eval_utils.eval_split(model.decoder, crit, loader, evaluation_args)

    print('loss: ', loss)
    if lang_stats:
        print(lang_stats)

    if opt.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open('vis/vis.json', 'w'))
