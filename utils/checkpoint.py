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
