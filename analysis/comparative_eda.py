#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare EDA features between two files

Usage:
    python compare_eda.py <inf1> <inf2> [-v <verbose>] [-c <column>]
"""

import sys
import re
import random
from argparse import ArgumentParser
from typing import List, Dict, Optional, Union

import pandas as pd
import plotext as plt # https://github.com/piccolomo/plotext

# import seaborn as sns
# import matplotlib.pyplot as plt

from eda import do_eda

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inf1', type=str, help='Input file 1')
    parser.add_argument('inf2', type=str, help='Input file 2')
    parser.add_argument('-v', '--verbose', type=int, required=False, default=0)
    parser.add_argument('-c', '--column', type=str, required=False, default='ttr')
    return parser.parse_args()

if __name__ == '__main__':  
    args = parse_args()
    df1 = do_eda(args.inf1, verbose=args.verbose)
    df2 = do_eda(args.inf2, verbose=args.verbose)

    plt.simple_multiple_bar(df1.index.to_list(), [df1[args.column].to_list(), df2[args.column].to_list()], labels=["file_1", "file_2"], width=100)
    plt.show()

    # df_diff = df1 - df2
    # df_comp = df1.compare(df2, keep_equal=True, result_names=('f1', 'f2'))
    # df3 = pd.concat([df_comp, df_diff], axis=1)
    # print(df3)

    # plt.bar(df3.index.to_list(), df3['ttr'].to_list(), orientation = 'h', width = 80) # flips index order
    # plt.simple_bar(df3.index.to_list(), df3['ttr'].to_list(), width = 80) # not compat with diverging bars

