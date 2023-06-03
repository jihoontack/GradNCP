import torch

from train.gradient_based import inner_adapt, inner_adapt_test_scale
from utils import MetricLogger, psnr, get_meta_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    filename_with_today_date = True
    return filename_with_today_date


def test_model(P, wrapper, loader, steps, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = wrapper.training
    wrapper.eval()
    wrapper.coord_init()

    for n, task_data in enumerate(loader):
        if n * P.test_batch_size > P.max_test_task:
            break

        task_data = {k: v.to(device, non_blocking=True) for k, v in task_data.items()}
        batch_size, context = get_meta_batch(P, task_data)

        params, loss_in = inner_adapt(
            wrapper,
            context,
            P.inner_lr,
            P.inner_steps_test,
            first_order=True,
        )
        psnr_in = psnr(loss_in)

        params_tt_gradscale, loss_in_tt_gradscale = inner_adapt_test_scale(
            wrapper,
            context,
            P.inner_lr,
            P.inner_steps_test,
            first_order=True,
            sample_type=P.sample_type,
            scale_type='grad'
        )
        psnr_in_tt_gradscale = psnr(loss_in_tt_gradscale)

        """ outer loss_out aggregate """
        with torch.no_grad():
            loss_out = wrapper(
                context,
                params=params
            )
            psnr_out = psnr(loss_out)
            loss_out_tt_gradscale = wrapper(
                context,
                params=params_tt_gradscale
            )
            psnr_out_tt_gradscale = psnr(loss_out_tt_gradscale)

        metric_logger.meters['loss_in'].update(loss_in.mean().item(), n=batch_size)
        metric_logger.meters['loss_out'].update(loss_out.mean().item(), n=batch_size)
        metric_logger.meters['psnr_in'].update(psnr_in.mean().item(), n=batch_size)
        metric_logger.meters['psnr_out'].update(psnr_out.mean().item(), n=batch_size)
        metric_logger.meters['loss_in_tt_gradscale'].update(loss_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['loss_out_tt_gradscale'].update(loss_out_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_in_tt_gradscale'].update(psnr_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_out_tt_gradscale'].update(psnr_out_tt_gradscale.mean().item(), n=batch_size)

        if n * P.test_batch_size > P.max_test_task:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    log_(' * [EVAL] [LossIn %.3f] [LossOut %.3f] [PSNRIn %.3f] [PSNROut %.3f]' %
         (metric_logger.loss_in.global_avg, metric_logger.loss_out.global_avg,
          metric_logger.psnr_in.global_avg, metric_logger.psnr_out.global_avg))
    log_(' * [EVAL Gradscale TT] [LossInGSTT %.3f] [LossOutGSTT %.3f] [PSNRInGSTT %.3f] [PSNROutGSTT %.3f]' %
         (metric_logger.loss_in_tt_gradscale.global_avg, metric_logger.loss_out_tt_gradscale.global_avg,
          metric_logger.psnr_in_tt_gradscale.global_avg, metric_logger.psnr_out_tt_gradscale.global_avg))

    if logger is not None:
        logger.scalar_summary('eval/loss_in', metric_logger.loss_in.global_avg, steps)
        logger.scalar_summary('eval/loss_out', metric_logger.loss_out.global_avg, steps)
        logger.scalar_summary('eval/psnr_in', metric_logger.psnr_in.global_avg, steps)
        logger.scalar_summary('eval/psnr_out', metric_logger.psnr_out.global_avg, steps)

        logger.scalar_summary('eval/loss_in_tt_gradscale', metric_logger.loss_in_tt_gradscale.global_avg, steps)
        logger.scalar_summary('eval/loss_out_tt_gradscale', metric_logger.loss_out_tt_gradscale.global_avg, steps)
        logger.scalar_summary('eval/psnr_in_tt_gradscale', metric_logger.psnr_in_tt_gradscale.global_avg, steps)
        logger.scalar_summary('eval/psnr_out_tt_gradscale', metric_logger.psnr_out_tt_gradscale.global_avg, steps)

    wrapper.train(mode)

    return max(metric_logger.psnr_out.global_avg, metric_logger.psnr_out_tt_gradscale.global_avg)
