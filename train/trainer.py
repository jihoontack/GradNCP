import time

import torch

from common.utils import is_resume
from utils import MetricLogger, save_checkpoint, save_checkpoint_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger):
    kwargs = {}
    kwargs_test = {}

    metric_logger = MetricLogger(delimiter="  ")

    """ resume option """
    is_best, start_step, best, psnr = is_resume(P, model, optimizer)

    """ training start """
    logger.log_dirname(f"Start training")

    for it, train_batch in enumerate(train_loader):
        step = start_step + it + 1

        if step > P.outer_steps:
            break

        model.iter = step  # update iteration in the model for adaptive ray sampling
        stime = time.time()
        train_batch = {k: v.to(device, non_blocking=True) for k, v in train_batch.items()}
        metric_logger.meters['data_time'].update(time.time() - stime)

        train_func(P, step, model, optimizer, train_batch,
                   metric_logger=metric_logger, logger=logger, **kwargs)

        """ evaluation & save the best model """
        if step % P.eval_step == 0:
            psnr = test_func(P, model, test_loader, step, logger=logger, **kwargs_test)

            if best < psnr:
                best = psnr
                save_checkpoint(P, step, best, model, optimizer.state_dict(),
                                logger.logdir, is_best=True, data_parallel=P.data_parallel)

            logger.scalar_summary('eval/best', best, step)
            logger.log('[EVAL] [Step %3d] [PSNR %5.2f] [Best %5.2f]' % (step, psnr, best))

        """ save model per save_step steps"""
        if step % P.save_step == 0:
            save_checkpoint_step(P, step, best, model, optimizer.state_dict(),
                                 logger.logdir, data_parallel=P.data_parallel)

    """ save last model"""
    save_checkpoint(P, P.outer_steps, best, model, optimizer.state_dict(),
                    logger.logdir, data_parallel=P.data_parallel)
