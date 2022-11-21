# -*- coding: utf-8 -*-
# @Time   : 2021/05/24
# @Author : enoche
# @Email  : enoche.chow@gmail.com

r"""
Main entry
"""

import argparse
import os
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '64'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='IGT', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='toy', help='name of datasets')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict)

