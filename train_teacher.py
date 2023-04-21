"""
Training teacher model.
This code differs from its origin, train.py in AoANet in these ways:
1. Using images rather than out-of-box feature data as input;
2. 'start_from' option is banned;
3. 'use_box' option must be false;
4. DataParallel is removed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import skimage.io

import adapter
from utils.loader import DataloaderWrapper
from models.encoder import ResNet101Encoder
import aoanet.opts as opts
import aoanet.eval_utils as eval_utils
import aoanet.misc.utils as utils
from aoanet.misc.rewards import init_scorer
from aoanet.misc.loss_wrapper import LossWrapper
from aoanet.models.AoAModel import AoAModel

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

dataset_root = '/root/autodl-tmp/datasets/MSCOCO'
device = 'cuda'


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


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


def train(opt):
    if opt.start_from is not None:
        raise ValueError()
    if opt.use_box:
        raise ValueError()
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    acc_steps = getattr(opt, 'acc_steps', 1)

    encoder = ResNet101Encoder(7)
    loader = DataloaderWrapper(encoder, dataset_root, opt, device=device)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {
        'iter': 0,
        'epoch': 0,
        'iterators': loader.iterators,
        'split_ix': loader.split_ix,
        'vocab': loader.get_vocab(),
        'opt': opt
    }
    histories = {}
    iteration = 0
    epoch = 0

    val_result_history = {}
    loss_history = {}
    lr_history = {}
    ss_prob_history = {}

    best_val_score = None
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = loader.get_vocab()
    decoder = AoAModel(opt).to(device)
    del opt.vocab
    lw_decoder = LossWrapper(decoder, opt)
    model = nn.ModuleDict({
        'endoer': encoder,
        'decoder': lw_decoder
    })

    epoch_done = True
    sc_flag = False
    encoder.train()
    lw_decoder.train()

    optimizer = optim.Adam(
        [{'params': decoder.parameters()}, {'params': encoder.parameters()}],
        opt.learning_rate,
        (opt.optim_alpha, opt.optim_beta),
        opt.optim_epsilon,
        opt.weight_decay
    )

    try:
        while True:
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    decoder.ss_prob = opt.ss_prob

                # If start self-critical training
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False

                epoch_done = False

            if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            # data = loader.get_batch('train', device=device)
            data = loader.get_batch('train')

            if iteration % acc_steps == 0:
                optimizer.zero_grad()
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            model_out = lw_decoder(
                data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'],
                data['gts'], torch.arange(0, len(data['gts'])), sc_flag
            )
            loss = model_out['loss'].mean()
            loss_sp = loss / acc_steps
            loss_sp.backward()

            if (iteration + 1) % acc_steps == 0:
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            train_loss = loss.item()
            end = time.time()
            if not sc_flag and iteration % 100 == 0:
                print(
                    "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(iteration, epoch, train_loss, end - start)
                )
            elif iteration % 100 == 0:
                print(
                    "iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}"
                    .format(iteration, epoch, model_out['reward'].mean(), end - start)
                )

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if iteration % opt.losses_log_every == 0:
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', decoder.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean(), iteration)

                loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = decoder.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix

            # make evaluation on validation set, and save model
            if iteration % opt.save_checkpoint_every == 0:
                # eval model
                eval_kwargs = {
                    'split': 'val',
                    'dataset': opt.input_json
                }
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    decoder, lw_decoder.crit, loader, eval_kwargs)

                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k, v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

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
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                save_checkpoint(model, infos, optimizer, opt, histories)
                if opt.save_history_ckpt:
                    save_checkpoint(model, infos, optimizer, opt, append=str(iteration))

                if best_flag:
                    save_checkpoint(model, infos, optimizer, opt, append='best')

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs != -1:
                break
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        save_checkpoint(model, infos, optimizer, opt)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


if __name__ == '__main__':
    opt = opts.parse_opt()
    train(opt)
