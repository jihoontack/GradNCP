import torch
from torch.utils.data import DataLoader

from common.args import parse_args
from common.utils import load_model
from data.dataset import get_dataset
from models.model import get_model
from utils import set_random_seed


def main():
    """ argument define """
    P = parse_args()
    P.rank = 0

    """ set torch device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    P.world_size = torch.cuda.device_count()
    P.data_parallel = P.world_size > 1
    assert not P.data_parallel  # no multi GPU

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    kwargs = {'batch_size': P.test_batch_size, 'shuffle': False,
              'pin_memory': True, 'num_workers': 4}
    test_set = get_dataset(P, dataset=P.dataset, only_test=True)
    test_loader = DataLoader(test_set, **kwargs)

    """ Initialize model """
    model = get_model(P).to(device)
    load_model(P, model)

    """ define train and test type """
    from evals import setup as test_setup
    test_func = test_setup(P.mode, P)

    """ test """
    test_func(P, model, test_loader, 0.0, logger=None)


if __name__ == "__main__":
    main()
