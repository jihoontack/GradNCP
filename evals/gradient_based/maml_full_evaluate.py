import torch
import torch.nn.functional as F

import lpips
from pytorch_msssim import ms_ssim, ssim

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

    lpips_score = lpips.LPIPS(net='alex').to(device)

    kwargs = {}
    if P.mode == 'maml_full_evaluate_gradscale':
        adapt = inner_adapt_test_scale
        kwargs['sample_type'] = P.sample_type
        kwargs['scale_type'] = 'grad'
    else:
        adapt = inner_adapt

    for n, task_data in enumerate(loader):
        task_data = {k: v.to(device, non_blocking=True) for k, v in task_data.items()}
        
        batch_size, context = get_meta_batch(P, task_data)
        params = adapt(
            wrapper,
            context,
            P.inner_lr,
            P.inner_steps_test,
            first_order=True,
            **kwargs
        )[0]
        with torch.no_grad():
            pred = wrapper(None, params).clamp(0, 1)

        if P.data_type == 'img':
            context = context[0]
            lpips_result = lpips_score((pred * 2 - 1), (context * 2 - 1)).mean()
            psnr_result = psnr(F.mse_loss(
                context.view(batch_size, -1), pred.view(batch_size, -1), reduce=False
            ).mean(dim=1)).mean()
            ms_ssim_result = ms_ssim(pred, context, data_range=1.0).mean()
            log_ms_ssim_result = (-10. * torch.log10(1 - ms_ssim(pred, context, data_range=1.0) + 1e-24)).mean()
            ssim_result = ssim(pred, context, data_range=1.0).mean()
            log_ssim_result = (-10. * torch.log10(1 - ssim(pred, context, data_range=1.0) + 1e-24)).mean()

        else:
            raise NotImplementedError()

        metric_logger.meters['lpips_result'].update(lpips_result.item(), n=batch_size)
        metric_logger.meters['psnr_result'].update(psnr_result.item(), n=batch_size)
        metric_logger.meters['ms_ssim_result'].update(ms_ssim_result.item(), n=batch_size)
        metric_logger.meters['ssim_result'].update(ssim_result.item(), n=batch_size)
        metric_logger.meters['log_ms_ssim_result'].update(log_ms_ssim_result.item(), n=batch_size)
        metric_logger.meters['log_ssim_result'].update(log_ssim_result.item(), n=batch_size)

        if n % 10 == 0:
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()

            log_(f' * [EVAL {n}] [PSNR %.3f] [LOG MS-SSIM %.3f] [LPIPS %.3f] '
                 '[SSIM %.3f] [MS-SSIM %.3f] [LOG SSIM %.3f] ' %
                 (metric_logger.psnr_result.global_avg, metric_logger.log_ms_ssim_result.global_avg,
                  metric_logger.lpips_result.global_avg, metric_logger.ssim_result.global_avg,
                  metric_logger.ms_ssim_result.global_avg, metric_logger.log_ssim_result.global_avg))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    log_(' * [EVAL] [PSNR %.3f] [LOG MS-SSIM %.3f] [LPIPS %.3f] '
         '[SSIM %.3f] [MS-SSIM %.3f] [LOG SSIM %.3f] ' %
         (metric_logger.psnr_result.global_avg, metric_logger.log_ms_ssim_result.global_avg,
          metric_logger.lpips_result.global_avg, metric_logger.ssim_result.global_avg,
          metric_logger.ms_ssim_result.global_avg, metric_logger.log_ssim_result.global_avg))

    wrapper.train(mode)
    torch.cuda.empty_cache()

    return metric_logger.psnr_result.global_avg
