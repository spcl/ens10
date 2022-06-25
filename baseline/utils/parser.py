# 1. Argument parser (todo: dataclass)

import argparse


def args_parser():

    parser = argparse.ArgumentParser(description='Train a model on ENS10')

    parser.add_argument('--loss', type=str, default='CRPS',
                     choices=['CRPS', 'L2'],
                     help='Loss function for training (default: CRPS)')

    parser.add_argument('--seed', type=int, default=16,
                     help='Torch Seed (default: 16)')

    parser.add_argument('--model', type=str, default='UNet',
                     choices=['UNet', 'MLP', 'EMOS'],
                     help='Model Architecture (default: UNet)')

    parser.add_argument('--ens-num', type=int, default=10,
                     help='Ensemble Number. '
                          'This is important for EMOS model (default: 10).')

    parser.add_argument('--data-path', type=str, default='./',
                     help='The path for both ENS10 and ERA5 datasets (default: ./)')

    parser.add_argument('--target-var', type=str, default='z500',
                     choices=['z500', 't850', 't2m'],
                     help='Target variable for prediction (default: z500)')

    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    parser.add_argument('--lr', '-lr', default=1e-2, type=float,
                        help='Learning rate (default: 1e-2)')

    parser.add_argument('--epochs', type=int , default=10,
                        help='Epochs (default: 10)')

    parser.add_argument('--batch-size', '-b', type=int , default=1,
                        help='Batch size (default: 1)')

    parser.add_argument('--make-plot', action='store_true',
                        help='make scatter plot for ens10 and era5')
    args = parser.parse_args()

    return args

