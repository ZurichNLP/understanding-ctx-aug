#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the similarity between src and tgt texts given the existance of a specific feature (e.g. question mark).

Usage:
    python compare_src_tgt_features.py <src_file> <tgt_file>
"""

import sys
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, cdist, dice, hamming, jaccard

from sim_metrics import simple_matching_coefficient, compute_sim

def iter_lines(file):    
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            yield line.strip()

if __name__ == '__main__':
    src_file = sys.argv[1]
    tgt_file = sys.argv[2]

    # breakpoint()
    src_counter = []
    tgt_counter = []
    for src_line, tgt_line in zip(iter_lines(src_file), iter_lines(tgt_file)):
        src_line = src_line.strip()
        tgt_line = tgt_line.strip()
        
        src_counter.append(sum([1 for x in src_line if x == '?']))
        tgt_counter.append(sum([1 for x in tgt_line if x == '?']))
        # breakpoint()
        
    # print(np.mean(src_counter))
    # print(np.mean(tgt_counter))

    identity_tgt = [1 if x > 0 else 0 for x in tgt_counter]
    for i in range(1, max(src_counter) + 1):
        identity_src = [1 if x >= i else 0 for x in src_counter]
        # identity_src = [1 if x == i else 0 for x in src_counter]
        compute_sim(identity_src, identity_tgt)        