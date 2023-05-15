#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: T. Kew

# Example usage:
#   python check_experiment_results.py KGD test_freq-bart_small

import sys
from pathlib import Path
from constants import *
from pprint import pprint

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def lc(file_path: Path):
    return sum(1 for _ in open(file_path, 'r', encoding='utf-8'))

def parse_file_name(file_name: Path):
    file_name = file_name.stem.split('-')
    return '-'.join(file_name[:2])

if __name__ == '__main__':

    tgt_model, tgt_dir = sys.argv[1:] # e.g. KGD test_freq-public_models
    
    seeds = [23, 42, 1984]

    # for seed in seeds:
    dirs_to_check = [f'resources/models/seed_{seed}/{tgt_model}/results' for seed in seeds]

    if tgt_model == 'KGD':
        experiments = list(kgd_experiment_configs.keys())
    elif tgt_model == 'CSD':
        experiments = list(csd_experiment_configs.keys())
    elif tgt_model == 'DD':
        experiments = list(dd_experiment_configs.keys())

    experiments.insert(0, 'baseline')

    summary = {}
    expected_files = set()
    max_lines = 0
    col_width = 0

    for dir_path in dirs_to_check:
        dir_path = Path(dir_path) / tgt_dir
        print(f'{bcolors.HEADER}{dir_path}{bcolors.ENDC}\n')
        if not dir_path.exists():
            raise RuntimeError(f"Directory {dir_path} does not exist!")
        
        
        for experiment in experiments:
            summary[experiment] = {}        
            files = sorted(list(dir_path.glob(f'*-{experiment}.csv')))

            for f in files:
                summary[experiment][parse_file_name(f)] = lc(f) 
                expected_files.add(parse_file_name(f))
                max_lines = summary[experiment][parse_file_name(f)]
                col_width = max(col_width, len(parse_file_name(f)))

        print(f'Models: {expected_files}\n')

        for experiment in summary:

            if len(summary[experiment]) < len(expected_files):
                print(
                    f'{bcolors.UNDERLINE}{experiment:{col_width+ 8}}{bcolors.ENDC}\t\t' \
                    f'{bcolors.WARNING}Missing {len(expected_files) - len(summary[experiment])} models: {bcolors.ENDC}' \
                    f'{bcolors.ENDC} {", ".join(list(expected_files - set(summary[experiment].keys())))} {bcolors.ENDC}'
                    )    
            else:
                print(f'{bcolors.ENDC}{experiment:{col_width + 8}}{bcolors.ENDC}\t\t{bcolors.OKGREEN}Has all models{bcolors.ENDC}')

            for exp_id, lines in summary[experiment].items():
                if lines != max_lines:
                    print(f'\t\t\t {bcolors.FAIL} {exp_id} only has {lines-1} result(s) {bcolors.ENDC}')
                    continue
        
        x = input('\nPress enter to continue...\n')
        if x == 'q':
            break

