"""
Training teacher model.
This code differs from its origin, train.py in AoANet in these ways:
1. Using images rather than out-of-box feature data as input;
2. 'use_box' option must be false;
3. DataParallel is removed;
4. self-critical is banned.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import traceback

import torch
import torch.optim as optim

import adapter
import utils
import models
import corrupter
import aoanet.misc.utils as aoa_utils
from aoanet.misc.loss_wrapper import LossWrapper


def train(opt):
    if opt.use_box:
        raise ValueError()
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = aoa_utils.if_use_feat(opt.caption_model)
    acc_steps = getattr(opt, 'acc_steps', 1)
    dataset_root = opt.dataset_root
    device = opt.training_device
    pre_trained_path = opt.pretrained_path

    encoder = models.ResNet101Encoder(7)
    loader = utils.DataloaderWrapper(encoder, dataset_root, opt, device=device)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    logger = utils.TensorBoardLogger(opt.checkpoint_path)

    if opt.start_from is not None:
        infos, histories = utils.load_record(opt)
    else:
        infos = {
            'iter': 0,
            'epoch': 0,
            'iterators': loader.iterators,
            'split_ix': loader.split_ix,
            'vocab': loader.get_vocab()
        }
        histories = {
            'val_result_history': {},
            'loss_history': {},
            'lr_history': {},
            'ss_prob_history': {}
        }
    infos['opt'] = opt

    eval_kwargs = {
        'split': 'val',
        'dataset': opt.input_json
    }

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    epoch_done = True
    sc_flag = False

    best_val_score = None
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = loader.get_vocab()
    decoder = models.AoAModelWrapper(opt, 1.0)
    del opt.vocab
    lw_decoder = LossWrapper(decoder, opt)
    crrupter = corrupter.get_instance(opt.corrupter)
    model = models.FixedFeatureCaptionModel(encoder, decoder, eval_kwargs)
    if opt.start_from is not None:
        utils.load_model(model, opt)
    model = model.to(device)

    optimizer = optim.Adam(
        [{'params': decoder.parameters()}, {'params': encoder.parameters()}],
        opt.learning_rate,
        (opt.optim_alpha, opt.optim_beta),
        opt.optim_epsilon,
        opt.weight_decay
    )
    if opt.start_from is not None:
        utils.load_optimizer(optimizer, opt)

    if pre_trained_path:
        model.load_state_dict(torch.load(os.path.join(pre_trained_path, 'model-best.pth')))
    model.train()

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
                    aoa_utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    decoder.ss_prob = opt.ss_prob

                epoch_done = False

            if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                aoa_utils.set_lr(optimizer, opt.current_lr)
            # data = loader.get_batch('train', device=device)
            data = loader.get_batch('train', corrupter=crrupter)

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
                aoa_utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            train_loss = loss.item()
            end = time.time()
            if not sc_flag and iteration % 1000 == 0:
                print(
                    "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(iteration, epoch, train_loss, end - start)
                )
            elif iteration % 1000 == 0:
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
                logger.train_log('train_loss', train_loss, iteration)
                logger.train_log('learning_rate', opt.current_lr, iteration)
                logger.train_log('scheduled_sampling_prob', decoder.ss_prob, iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = decoder.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix

            # make evaluation on validation set, and save model
            if iteration % opt.save_checkpoint_every == 0:
                eval_kwargs.update(vars(opt))
                model.eval()
                best_val_score = utils.checkpoint(
                    model, optimizer, lw_decoder.crit, loader,
                    eval_kwargs, histories, infos, opt,
                    iteration, best_val_score,
                    logger
                )
                model.train()

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs != -1:
                break
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(model, infos, optimizer, opt)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


if __name__ == '__main__':
    parser = utils.set_parser()
    opt = parser.parse_args()
    train(opt)
