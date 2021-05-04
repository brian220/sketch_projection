#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys

# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net
from core.evaluate import evaluate_net
from core.test_opt_gd import test_opt_gd
from core.test_opt_network import test_opt_net
# from core.evaluate_graphx import evaluate_net
# from core.evaluate_graphx_fixed_view import evaluate_fixed_view_net
from core.evaluate_hand_draw import evaluate_hand_draw_net
from core.evaluate_multi_view import evaluate_multi_view_net

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--train', dest='train', help='Train neural networks', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--evaluate', dest='evaluate', help='Evaluate neural networks', action='store_true')
    parser.add_argument('--evaluate_hand_draw', dest='evaluate_hand_draw', help='Evaluate neural networks by hand draw sketch', action='store_true')
    parser.add_argument('--evaluate_fixed_view', dest='evaluate_fixed_view', help='Evaluate neural networks in fixed views', action='store_true')
    parser.add_argument('--evaluate_multi_view', dest='evaluate_multi_view', help='Evaluate neural networks in multi views', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--test_opt_gd', dest='test_opt_gd', help='Test time optimization by gradient descent', action='store_true')
    parser.add_argument('--test_opt_net', dest='test_opt_net', help='Test time optimization by network fine-tune', action='store_true')
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path

    # Print config
    print('Use config:')
    pprint(cfg)
    
    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if args.train:
        train_net(cfg)
    elif args.test:
        test_net(cfg)
    elif args.evaluate:
        evaluate_net(cfg)
    elif args.evaluate_hand_draw:
        evaluate_hand_draw_net(cfg)
    elif args.evaluate_multi_view:
        evaluate_multi_view_net(cfg)
    elif args.test_opt_gd:
        test_opt_gd(cfg)
    elif args.test_opt_net:
        test_opt_net(cfg)
    else:
        print("Please specify the arguments (--train, --test, --evaluate)")
    """
    elif args.evaluate:
        evaluate_net(cfg)
    elif args.evaluate_hand_draw:
        evaluate_hand_draw_net(cfg)
    elif args.evaluate_fixed_view:
        evaluate_fixed_view_net(cfg)
    """

if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
