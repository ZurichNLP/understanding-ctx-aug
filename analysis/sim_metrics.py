#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def simple_matching_coefficient(X, Y):
    """
    Simple Matching Coefficient
    """
    return np.sum(X == Y) / len(X)

if __name__ == '__main__':
    p = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    q = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])

    print(simple_matching_coefficient(p, q))
    print(simple_matching_coefficient(q, p))
