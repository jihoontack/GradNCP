import time

import torch

from train.gradient_based import inner_adapt
from utils import psnr, get_meta_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    filename_with_today_date = True
    return filename_with_today_date


def param_consistency(P, params, params_bootstrap, bs):
    param_norm = []
    for (name, param) in params.items():
        updated_param = params_bootstrap[name].detach() - param
        updated_param = updated_param.view(bs, -1)
        param_norm.append(torch.norm(updated_param, p=2, dim=1, keepdim=True))
    return torch.norm(torch.cat(param_norm, dim=1), p=2, dim=1).mean()


def train_step(P, steps, wrapper, optimizer, task_data, metric_logger, logger):

    stime = time.time()
    wrapper.train()

    batch_size, context = get_meta_batch(P, task_data)

    wrapper.support = True
    params, loss_in = inner_adapt(
        wrapper,
        context,
        P.inner_lr,
        P.inner_steps,
        first_order=P.mode == 'fomaml',
        sample_type=P.sample_type,
    )

    wrapper.coord_init()
    wrapper.support = False
    params_boot, loss_in_boot = inner_adapt(
        wrapper,
        context,
        P.inner_lr_boot,
        P.inner_steps_boot,
        first_order=True,
        params=params,
    )

    """ outer loss aggregate """
    wrapper.coord_init()
    loss_out = wrapper(
        context,
        params=params
    )

    loss_boot = P.lam * param_consistency(P, params, params_boot, batch_size)
    loss = loss_out.mean() + loss_boot

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapper.decoder.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()

    """ track stat """
    metric_logger.meters['batch_time'].update(time.time() - stime, n=batch_size)
    metric_logger.meters['loss_in'].update(loss_in.mean().item(), n=batch_size)
    metric_logger.meters['loss_out'].update(loss_out.mean().item(), n=batch_size)
    metric_logger.meters['psnr_in'].update(psnr(loss_in).mean().item(), n=batch_size)
    metric_logger.meters['psnr_out'].update(psnr(loss_out).mean().item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    if steps % P.print_step == 0:
        logger.log_dirname(f"Step {steps}")
        logger.scalar_summary('train/loss_in',
                              metric_logger.loss_in.global_avg, steps)
        logger.scalar_summary('train/loss_out',
                              metric_logger.loss_out.global_avg, steps)
        logger.scalar_summary('train/psnr_in',
                              metric_logger.psnr_in.global_avg, steps)
        logger.scalar_summary('train/psnr_out',
                              metric_logger.psnr_out.global_avg, steps)
        logger.scalar_summary('train/batch_time',
                              metric_logger.batch_time.global_avg, steps)

        logger.log('[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] '
                   '[LossIn %f] [LossOut %f] [PSNRIn %.3f] [PSNROut %.3f]' %
                   (steps, metric_logger.batch_time.global_avg, metric_logger.data_time.global_avg,
                    metric_logger.loss_in.global_avg, metric_logger.loss_out.global_avg,
                    metric_logger.psnr_in.global_avg, metric_logger.psnr_out.global_avg))

        metric_logger.reset()
