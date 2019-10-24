# -*- coding: utf-8 -*-
"""
Created on Sep 25 2019

@author: Matteo Toso'
"""
import argparse
from utils.config import *

__all__ = [
    'parse_parameters'
]


def _add_wiberg_parameters(parser, lr=5e-2, data_path=SH_DIR + "train_sh.h5", name='New_test',
                           epochs=20):
    parser.add_argument('--wlr', type=float, default=lr, help='Wiberg learning rate')
    parser.add_argument('--batch_size', type=float, default=4, help='images processed at the same time')
    parser.add_argument('--train', type=bool, default=False, help='flag to train the Wiberg weights')
    parser.add_argument('--epochs', type=int, default=epochs, help='number of training epochs')
    parser.add_argument('--metric', type=int, default=2, help='0) Proc 14, 1) Proc 17, 2) Euclid')
    parser.add_argument('--models', type=int, default=3, help='Number of models (3,6,10)')
    parser.add_argument('--input_path', type=str, default=data_path, help='Directory to data set')
    parser.add_argument('--check_path', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--save_f', type=int, default=100, help='frequency of checkpoints')
    parser.add_argument('--name', type=str, default=name, help='Output directory')
    parser.add_argument('--l2_solver', type=bool, default=False, help='Using L2 rather than Huber?')


def parse_parameters(*args, **kargs):
    parser = argparse.ArgumentParser()
    _add_wiberg_parameters(parser, *args, **kargs)
    FLAGS = parser.parse_args()
    return FLAGS

