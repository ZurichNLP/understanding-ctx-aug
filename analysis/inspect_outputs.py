#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple helper script to inspect a set of inputs, references and one or more model outputs.

Usage:
    python inspect_outputs.py <pred_files> [-s <src_file>] [-t <tgt_file>] [--seed <seed>]
"""

import os
import sys
import random
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pred_files', nargs='*', type=str, help='Predicted file')
    parser.add_argument('-s', '--src_file', type=str, required=False, default=None, help='Source file')
    parser.add_argument('-t', '--tgt_file', type=str, required=False, default=None, help='Target file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling')
    return parser.parse_args()

def read_lines(file):
    lines = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
    return lines

if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)

    src_lines = read_lines(args.src_file) if args.src_file else None
    tgt_lines = read_lines(args.tgt_file) if args.tgt_file else None
    pred_lines = [read_lines(pred_file) for pred_file in args.pred_files]

    # check that all files have the same number of lines
    if src_lines:
        len(pred_lines[0]) == len(src_lines)
    if tgt_lines:
        len(pred_lines[0]) == len(tgt_lines)
    if len(pred_lines) > 1:
        assert all(len(pred_lines[0]) == len(pred_lines[i]) for i in range(1, len(pred_lines)))
    
    random_ints = random.sample(range(len(pred_lines[0])), len(pred_lines[0]))

    for i in random_ints:
        os.system('clear')
        print()
        if src_lines:
            print(f'Source ({i}):\t\t{src_lines[i]}')
        if tgt_lines:
            print(f'Target ({i}):\t\t{tgt_lines[i]}')
        for j in range(len(pred_lines)):
            print(f'Preds file {j+1} ({i}):\t{pred_lines[j][i]}')
        
        action = input('\n\tPress q exit...')
        if action == 'q':
            break

