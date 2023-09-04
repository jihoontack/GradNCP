import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup(mode, P):

    fname = f'{P.dataset}_{P.decoder}_{mode}_bs{P.batch_size}_inner{P.inner_steps}'

    if mode in ['maml']:
        from train.gradient_based.maml import train_step
        from train.gradient_based.maml import check
    elif mode in ['maml_bootstrap_param']:
        from train.gradient_based.maml_boot import train_step
        from train.gradient_based.maml_boot import check
        fname += f'_ratio{P.data_ratio}_{P.sample_type}_L{P.inner_steps_boot}_lam{P.lam}'
        if P.inner_lr_boot is None:
            P.inner_lr_boot = P.inner_lr
    else:
        raise NotImplementedError()

    today = check(P)
    if P.no_date:
        today = False

    fname += f'_seed_{P.seed}'

    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train_step, fname, today
