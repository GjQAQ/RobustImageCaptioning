from __future__ import division, absolute_import, print_function

import os

import torch

from aoanet.misc import utils as utils


def save_checkpoint(model, infos, optimizer, opt, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' % append)
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'model_optimizer%s.pth' % append)
    torch.save(optimizer.state_dict(), optimizer_path)
    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '%s.pkl' % append), 'wb') as f:
        utils.pickle_dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '%s.pkl' % append), 'wb') as f:
            utils.pickle_dump(histories, f)


def load_record(opt):
    # open old infos and check if models are compatible
    with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
        infos = utils.pickle_load(f)
        saved_model_opt = infos['opt']
        need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                "Command line argument and saved model disagree on '%s' " % checkme

    if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories = utils.pickle_load(f)
    return infos, histories


def load_model(model, opt):
    # check if all necessary files exist
    assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
    assert os.path.isfile(os.path.join(opt.start_from, "infos_" + opt.id + ".pkl")), \
        "infos.pkl file does not exist in path %s" % opt.start_from
    model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))


def load_optimizer(optimizer, opt):
    optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'model_optimizer.pth')))
