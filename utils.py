import pickle
import random
import shutil
import sys
from datetime import datetime
import os
import time
from collections import OrderedDict, defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True, today=True, rank=0):
        self.rank = rank
        self.log_path = './logs/'
        self.logdir = None

        if self.rank == 0:
            if not os.path.exists(self.log_path):
                os.mkdir(self.log_path)
            self.today = today

            logdir = self._make_dir(fn)
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            if len(os.listdir(logdir)) != 0 and ask:
                ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)

            self.set_dir(logdir)

    def _make_dir(self, fn):
        if self.today:
            today = datetime.today().strftime("%y%m%d")
            logdir = self.log_path + today + '_' + fn
        else:
            logdir = self.log_path + fn
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def close_writer(self):
        if self.rank == 0:
            self.writer.close()

    def log(self, string):
        if self.rank == 0:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.rank == 0:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.rank == 0: self.writer.add_scalar(tag, value, step)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(logdir, mode='last'):
    model_path = os.path.join(logdir, f'{mode}.model')
    optim_path = os.path.join(logdir, f'{mode}.optim')
    config_path = os.path.join(logdir, f'{mode}.configs')
    lr_path = os.path.join(logdir, f'{mode}.lr')

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None, None

    if os.path.exists(lr_path):
        lr_dict = torch.load(lr_path)
    else:
        lr_dict = None

    return model_state, optim_state, cfg, lr_dict


def save_checkpoint(P, step, best, model, optim_state, logdir,
                    is_best=False, suffix='', data_parallel=False):
    if P.rank != 0:
        return None

    if is_best:
        prefix = 'best'
    else:
        prefix = 'last'

    if data_parallel:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    last_model = os.path.join(logdir, f'{prefix}{suffix}.model')
    last_optim = os.path.join(logdir, f'{prefix}{suffix}.optim')
    last_config = os.path.join(logdir, f'{prefix}{suffix}.configs')

    if isinstance(P.inner_lr, OrderedDict):
        last_lr = os.path.join(logdir, f'{prefix}{suffix}.lr')
        torch.save(P.inner_lr, last_lr)
    if hasattr(P, 'moving_average'):
        last_ema = os.path.join(logdir, f'{prefix}{suffix}.ema')
        torch.save(P.moving_average, last_ema)
    if hasattr(P, 'moving_inner_lr'):
        last_lr_ema = os.path.join(logdir, f'{prefix}{suffix}.lr_ema')
        torch.save(P.moving_inner_lr, last_lr_ema)

    opt = {
        'step': step,
        'best': best
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_checkpoint_step(P, step, best, model, optim_state,
                         logdir, suffix='', data_parallel=False):

    if P.rank != 0:
        return None

    if data_parallel:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    last_model = os.path.join(logdir, f'step{step}{suffix}.model')
    last_optim = os.path.join(logdir, f'step{step}{suffix}.optim')
    last_config = os.path.join(logdir, f'step{step}{suffix}.configs')

    if isinstance(P.inner_lr, OrderedDict):
        last_lr = os.path.join(logdir, f'step{step}{suffix}.lr')
        torch.save(P.inner_lr, last_lr)
    if hasattr(P, 'moving_average'):
        last_ema = os.path.join(logdir, f'step{step}{suffix}.ema')
        torch.save(P.moving_average, last_ema)
    if hasattr(P, 'moving_inner_lr'):
        last_lr_ema = os.path.join(logdir, f'step{step}{suffix}.lr_ema')
        torch.save(P.moving_inner_lr, last_lr_ema)

    opt = {
        'step': step,
        'best': best
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cycle(loader):
    while True:
        for x in loader:
            yield x


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def psnr(mse):
    return -10.0 * torch.log10(mse+1e-24)


def get_meta_batch(P, task_data):

    if P.data_type == 'img':
        batch_size = task_data['imgs'].size(0)
        context = [task_data['imgs']]
    else:
        raise NotImplementedError()
    
    return batch_size, context
