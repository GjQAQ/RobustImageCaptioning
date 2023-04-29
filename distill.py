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


def distill(opt):
    if opt.use_box:
        raise ValueError()
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = aoa_utils.if_use_feat(opt.caption_model)
    acc_steps = getattr(opt, 'acc_steps', 1)
    dataset_root = opt.dataset_root
    device = opt.training_device

    student_encoder = models.ResNet101Encoder(7)
    loader = utils.DataloaderWrapper(student_encoder, dataset_root, opt, device=device)
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

    best_val_score = None
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = loader.get_vocab()
    student_decoder = models.AoAModelWrapper(opt)
    teacher_encoder = models.ResNet101Encoder(7)
    teacher_decoder = models.AoAModelWrapper(opt)
    del opt.vocab

    eval_kwargs.update(vars(opt))
    student = models.FixedFeatureCaptionModel(student_encoder, student_decoder, eval_kwargs)
    teacher = models.FixedFeatureCaptionModel(teacher_encoder, teacher_decoder, eval_kwargs)
    crrupter = corrupter.get_instance(opt.corrupter)
    container = models.DistillationContainer(
        teacher, student, crrupter, opt.distilling_temperature, smoothing=opt.label_smoothing
    )
    if opt.start_from is not None:
        utils.load_model(student, opt)
    teacher.load_state_dict(torch.load(opt.teacher_checkpoint))
    student.load_state_dict(torch.load(opt.teacher_checkpoint))
    student.to(device)
    teacher.to(device)

    optimizer = optim.Adam(
        [{'params': student_decoder.parameters()}, {'params': student_encoder.parameters()}],
        opt.learning_rate,
        (opt.optim_alpha, opt.optim_beta),
        opt.optim_epsilon,
        opt.weight_decay
    )
    if opt.start_from is not None:
        utils.load_optimizer(optimizer, opt)

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
                    student_decoder.ss_prob = opt.ss_prob

                epoch_done = False

            if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                aoa_utils.set_lr(optimizer, opt.current_lr)
            # data = loader.get_batch('train', device=device)
            data = loader.get_batch('train', image_only=True)

            if iteration % acc_steps == 0:
                optimizer.zero_grad()
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            loss, soft_loss, hard_loss = container.forward(data['images'], data['labels'], data['masks'])
            loss.backward()

            if (iteration + 1) % acc_steps == 0:
                aoa_utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            train_loss = loss.item()
            end = time.time()
            if iteration % 1000 == 0:
                print(
                    "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(iteration, epoch, train_loss, end - start)
                )

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if iteration % opt.losses_log_every == 0:
                logger.train_log('train_loss', train_loss, iteration)
                logger.train_log('soft_loss', soft_loss, iteration)
                logger.train_log('hard_loss', hard_loss, iteration)
                logger.train_log('learning_rate', opt.current_lr, iteration)
                logger.train_log('scheduled_sampling_prob', student_decoder.ss_prob, iteration)

                histories['loss_history'][iteration] = train_loss
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = student_decoder.ss_prob

            # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix

            # make evaluation on validation set, and save model
            if iteration % opt.save_checkpoint_every == 0:
                eval_kwargs.update(vars(opt))
                best_val_score = utils.checkpoint(
                    student, optimizer, aoa_utils.LanguageModelCriterion(), loader,
                    eval_kwargs, histories, infos, opt,
                    iteration, best_val_score,
                    logger
                )

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs != -1:
                break
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(student, infos, optimizer, opt)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


if __name__ == '__main__':
    parser = utils.set_distill_parser()
    opt = parser.parse_args()
    distill(opt)
