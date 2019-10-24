# -*- coding: utf-8 -*-
"""
Created on Mar 23 11:57 2017

@author: Denis Tome'
"""
import logging
from os.path import exists, dirname, abspath, join
import os

__all__ = [
    'H36M_NUM_JOINTS',
    'NUM_CAMERAS',
    'DATA_DIR',
    'CHK_DIR',
    'SH_DIR',
    'ACTIONS'
]

_logger = logging.getLogger('Config')

# net attributes
H36M_NUM_JOINTS = 17
NUM_CAMERAS = 4

ACTIONS = ["Directions", "Discussion", "Eating", "Greeting",
           "Phoning", "Photo", "Posing", "Purchases",
           "Sitting", "SittingDown", "Smoking", "Waiting",
           "WalkDog", "Walking", "WalkTogether"]

# path definitions
_CURR_DIR = dirname(abspath(__file__))

CHK_DIR = join(_CURR_DIR, '../results/')
if not exists(CHK_DIR):
    os.makedirs(CHK_DIR)

DATA_DIR = join(_CURR_DIR, '../data/')
if not exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SH_DIR = DATA_DIR
if not exists(SH_DIR):
    os.makedirs(DATA_DIR)
