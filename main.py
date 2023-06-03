import torch
from torch.utils.data import DataLoader

from common.args import parse_args
from common.utils import InfiniteSampler, get_optimizer, load_model
from data.dataset import get_dataset
from models.model import get_model
from train.trainer import meta_trainer
from utils import Logger, set_random_seed


def main(rank, P):
    P.rank = rank

    """ set torch device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset """
    train_set, test_set = get_dataset(P, dataset=P.dataset)

    """ define dataloader """
    kwargs = {'pin_memory': True, 'num_workers': 0}
    train_sampler = InfiniteSampler(train_set, rank=rank, num_replicas=1, shuffle=True, seed=P.seed)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=P.batch_size, num_workers=4, prefetch_factor=2)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

    """ Initialize model, optimizer """
    model = get_model(P).to(device)
    optimizer = get_optimizer(P, model)

    """ define train and test type """
    from train import setup as train_setup
    from evals import setup as test_setup
    train_func, fname, today = train_setup(P.mode, P)
    test_func = test_setup(P.mode, P)

    """ define logger """
    logger = Logger(fname, ask=P.resume_path is None, today=today, rank=P.rank)
    logger.log(P)
    logger.log(model)

    """ load model if necessary """
    load_model(P, model, logger)

    """ apply data parrallel for multi-gpu training """
    if P.data_parallel:
        raise NotImplementedError()  # Currently having some error with DP

    """ train """
    meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger)

    """ close tensorboard """
    logger.close_writer()


if __name__ == "__main__":
    """ argument define """
    P = parse_args()

    P.world_size = torch.cuda.device_count()
    P.data_parallel = P.world_size > 1

    # We use data parallel (DP) rather than distributed data parallel (DDP)
    # Currently, Meta-learning with DDP cause a problem, see below issues:
    # https://github.com/pytorch/pytorch/issues/47562
    # https://github.com/pytorch/pytorch/issues/48531
    # https://github.com/pytorch/pytorch/issues/63812
    # if P.distributed:
    #     os.environ["MASTER_ADDR"] = 'localhost'
    #     os.environ["MASTER_PORT"] = P.port
    #     mp.spawn(main, nprocs=P.world_size, args=(P,))
    # else:

    main(0, P)
