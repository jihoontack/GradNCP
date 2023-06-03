import os
from argparse import ArgumentParser
import yaml


def parse_args():
    """Command-line argument parser for train."""

    parser = ArgumentParser(
        description='Official PyTorch implementation of GradNCP'
    )

    parser.add_argument('--dataset', help='Dataset',
                        type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='maml', type=str)
    parser.add_argument("--seed", type=int,
                        default=0, help='random seed')
    parser.add_argument("--rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--distributed', help='automatically change to True for GPUs > 1',
                        default=False, type=bool)
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument('--configs', help='Path to the loading configs',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--eval_step', help='Epoch steps to compute accuracy/error',
                        default=1000, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save checkpoint',
                        default=50000, type=int)
    parser.add_argument('--print_step', help='Epoch steps to print/track training stat',
                        default=100, type=int)
    parser.add_argument("--no_date", help='do not save the date',
                        action='store_true')

    """ Training Configurations """
    parser.add_argument('--inner_steps', help='meta-learning inner-step',
                        default=3, type=int)
    parser.add_argument('--inner_steps_test', help='meta-learning inner-step at test-time',
                        default=3, type=int)
    parser.add_argument('--outer_steps', help='meta-learning outer-step',
                        default=100000, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--inner_lr', type=float, default=1e-2, metavar='INLR',
                        help='learning rate of inner gradients')
    parser.add_argument('--inner_lr_boot', type=float, default=None,
                        help='learning rate of inner gradients')
    parser.add_argument('--inner_steps_boot', type=int, default=5,
                        help='learning rate of inner gradients')
    parser.add_argument('--batch_size', help='Batch size',
                        default=16, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=25, type=int)
    parser.add_argument('--max_test_task', help='Max number of task for inference',
                        default=100, type=int)
    parser.add_argument('--lam', type=float, default=1.)

    """ Decoder Configurations """
    parser.add_argument('--inr', help='model type', type=str, default='siren')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_hidden', type=int, default=256)
    parser.add_argument('--dim_in', type=int, default=3)
    parser.add_argument('--dim_out', type=int, default=4)
    parser.add_argument('--w0', type=float, default=30.)

    # sample_type
    parser.add_argument("--data_ratio", help='sampling ratio',
                        default=0.25, type=float)
    parser.add_argument("--sample_type", help='sampling method',
                        default='none', type=str)

    args = parser.parse_args()
    if args.configs is not None and os.path.exists(args.configs):
        load_cfg(args)

    return args


def load_cfg(args):
    with open(args.configs, "rb") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        args.__dict__[key] = value

    return args
