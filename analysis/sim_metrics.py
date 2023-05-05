#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.spatial.distance import pdist, cdist, dice, hamming, jaccard

def simple_matching_coefficient(X, Y):
    """
    Simple Matching Coefficient
    """
    return np.sum(X == Y) / len(X)

def compute_sim(src_preds, tgt_preds):
    smi = simple_matching_coefficient(src_preds, tgt_preds)
    pearson = pearsonr(src_preds, tgt_preds)[0]
    jaccard_score = jaccard(src_preds, tgt_preds)

    data = {
        'AVG SRC': round(sum(src_preds)/len(src_preds), 4),
        'AVG TGT': round(sum(tgt_preds)/len(tgt_preds), 4),
        'SMI': round(smi, 4),
        'Pearson R': round(pearson, 4),
        'JAC': round(jaccard_score, 4)
    }

    return data

def compare_dist_metrics(X, Y):
    print('X:', X)
    print('Y:', Y)
    print('SMC:', simple_matching_coefficient(X, Y))
    print('SMC:', simple_matching_coefficient(Y, X))
    print('JAC:', jaccard(X, Y))
    print('JAC:', jaccard(Y, X))
    print('HAM:', hamming(X, Y))
    print('HAM:', hamming(Y, X))
    print('DIC:', dice(X, Y))
    print('DIC:', dice(Y, X))
    print()

if __name__ == '__main__':

    p = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    q = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    compare_dist_metrics(p, q)

    p = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    q = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    compare_dist_metrics(p, q)

    p = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    q = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    compare_dist_metrics(p, q)

    p = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    q = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    compare_dist_metrics(p, q)

    p = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    q = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    compare_dist_metrics(p, q)
