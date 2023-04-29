from __future__ import division, absolute_import, print_function

import os

import torch

import eval
import aoanet.misc.utils as misc_utils
import aoanet.eval_utils as eval_utils


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
        misc_utils.pickle_dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '%s.pkl' % append), 'wb') as f:
            misc_utils.pickle_dump(histories, f)


def load_record(opt):
    # open old infos and check if models are compatible
    with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
        infos = misc_utils.pickle_load(f)
        saved_model_opt = infos['opt']
        need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                "Command line argument and saved model disagree on '%s' " % checkme

    if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories = misc_utils.pickle_load(f)
    return infos, histories


def load_model(model, opt):
    # check if all necessary files exist
    assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
    assert os.path.isfile(os.path.join(opt.start_from, "infos_" + opt.id + ".pkl")), \
        "infos.pkl file does not exist in path %s" % opt.start_from
    model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))


def load_optimizer(optimizer, opt):
    optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'model_optimizer.pth')))


@torch.no_grad()
def checkpoint(
    model, optimizer, crit, loader,
    eval_kwargs, histories, infos, opt,
    iteration, best_val_score,
    logger
):
    # val_loss, predictions, lang_stats = eval_utils.eval_split(
    #     model.decoder, crit, loader, eval_kwargs)
    val_loss, predictions, lang_stats = eval.evaluate(model, crit, loader, eval_kwargs)

    # Write validation result into summary
    logger.val_log('validation_loss', val_loss, iteration)
    if lang_stats is not None:
        for k, v in lang_stats.items():
            logger.val_log(k, v, iteration)
    histories['val_result_history'][iteration] = {
        'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions
    }

    # Save model if improving on validation result
    if opt.language_eval == 1:
        current_score = lang_stats['CIDEr']
    else:
        current_score = - val_loss

    best_flag = False
    if best_val_score is None or current_score > best_val_score:
        best_val_score = current_score
        best_flag = True
    # Dump miscalleous information
    infos['best_val_score'] = best_val_score

    save_checkpoint(model, infos, optimizer, opt, histories)
    if opt.save_history_ckpt:
        save_checkpoint(model, infos, optimizer, opt, append=str(iteration))
    if best_flag:
        save_checkpoint(model, infos, optimizer, opt, append='best')

    return best_val_score
