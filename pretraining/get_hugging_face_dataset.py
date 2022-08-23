#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from datasets import load_dataset
from pathlib import Path

dataset_name = sys.argv[1] # "bookcorpus"
outpath = sys.argv[2] # "resources/bookcorpus"

dataset = load_dataset(dataset_name)
dataset.save_to_disk(outpath)
print('***')
print(dataset)
print(f'Saved Dataset to {outpath}')

raw_path = Path(outpath) / 'raw' / 'train.csv'
dataset['train'].to_csv(raw_path, header=None, index=False) # sed -i '1d' train.csv
print('***')
print(f'Saved `train` split as raw dataset to {raw_path} . Check that the first line is removed!')