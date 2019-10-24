# -*- coding: utf-8 -*-
"""
Created on Sep 20 11:43 2018

@author: Matteo Toso
"""

from utils.parameters_io import *
from utils.settings import parse_parameters
from utils.train import train
from utils.test import test


def main():
    exp_path = CHK_DIR+FLAGS.name
    build_dir_tree(exp_path)

    if FLAGS.train:
        print 'Starting training...'
        train()
    else:
        print 'Starting testing...'
        test()


if __name__ == "__main__":
    FLAGS = parse_parameters()
    main()
