import os

import numpy as np
import torch
import torch.optim as optim

from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(P, model):
    params = model.parameters()
    optimizer = optim.Adam(params, lr=P.lr)
    return optimizer


def is_resume(P, model, optimizer):
    if P.resume_path is not None:
        model_state, optim_state, config, lr_dict = load_checkpoint(P.resume_path, mode='best')
        model.load_state_dict(model_state, strict=not P.no_strict)
        optimizer.load_state_dict(optim_state)
        start_step = config['step']
        best = config['best']
        is_best = False
        psnr = 0.0
        if lr_dict is not None:
            P.inner_lr = lr_dict
    else:
        is_best = False
        start_step = 1
        best = 0.0
        psnr = 0.0
    return is_best, start_step, best, psnr


def load_model(P, model, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if P.load_path is not None:
        log_(f'Load model from {P.load_path}')
        checkpoint = torch.load(P.load_path)
        if P.rank != 0:
            model.__init_low_rank__(rank=P.rank)

        not_loaded = model.load_state_dict(checkpoint, strict=P.no_strict)
        print (not_loaded)

        if os.path.exists(P.load_path[:-5] + 'lr'):  # Meta-SGD
            log_(f'Load lr from {P.load_path[:-5]}lr')
            lr = torch.load(P.load_path[:-5] + 'lr')
            for (_, param) in lr.items():
                param.to(device)
            P.inner_lr = lr


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
