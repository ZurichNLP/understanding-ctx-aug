#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import random

src_file = sys.argv[1]
tgt_file = sys.argv[2]
gen_file = sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
random.seed(seed)

def read_lines(file):
    with open(file, 'r', encoding='utf8') as f:
        return f.readlines()

if __name__ == '__main__':
    src_lines = read_lines(src_file)
    tgt_lines = read_lines(tgt_file)
    gen_lines = read_lines(gen_file)

    assert len(src_lines) == len(tgt_lines) == len(gen_lines)

    random_ints = random.sample(range(len(src_lines)), len(src_lines))

    for i in random_ints:
        os.system('clear')
        print()
        print(f'Source ({i}):\t{src_lines[i]}')
        print(f'Target ({i}):\t{tgt_lines[i]}')
        print(f'Model ({i}):\t{gen_lines[i]}')
        
        action = input('\n\tPress q exit...')
        if action == 'q':
            break

