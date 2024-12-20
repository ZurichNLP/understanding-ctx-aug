#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example usage:

    python annotate_src_tgt_features.py \
        --dataset "resources/data/Topical-Chat/KGD/train.json" \
        --output_file "resources/data/Topical-Chat/KGD/kgd_train_src_tgt_question_counts.csv"

"""

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset

from evaluation.sentence_processing import count_questions

# set args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

def load_json_dataset(dataset: Path):   
    split = dataset.stem
    extension = dataset.suffix.strip('.')
    print(split, extension, str(dataset))
    dataset_dict = load_dataset(extension, data_files={'test': str(dataset)})
    return dataset_dict # concatenate_datasets(loaded_datasets)


def get_src_tgt_question_counts(example):
    src_tgt = example['turns'] + [example['target']]
    # print(src_tgt)
    result = count_questions(src_tgt, verbose=False).tolist()
    # print(result)
    example['src_tgt_question_counts'] = result
    return example

# write to csv
def write_features_to_file(dataset, outfile):
    with open(outfile, 'w', encoding='utf8') as f:
        for i, item in tqdm(enumerate(dataset)):
            f.write(f"{item['src_tgt_question_counts']}\n")
            if i < 5:
                print(item['src_tgt_question_counts'])
                
    print(f'Wrote {i+0} items to {outfile}')

    return

if __name__ == '__main__':
    args = parser.parse_args()

    dataset = load_json_dataset(Path(args.dataset))['test']
    print(f'Loaded {len(dataset)} examples from {args.dataset}')
    print('Example:')
    print(dataset[0])

    dataset = dataset.map(get_src_tgt_question_counts, batched=False, desc="Counting questions")

    write_features_to_file(dataset, args.output_file)

# dataset = load_json_dataset(Path('../resources/data/Commonsense-Dialogues/CSD/train.json'))['test']
# print(dataset)
# pprint(dataset[0])
# dataset = dataset.map(get_src_tgt_question_counts, batched=False, desc="Counting questions")
# print(dataset)
# print(dataset[0])
# write_features_to_file(dataset, '../resources/data/Commonsense-Dialogues/CSD/train_src_tgt_question_counts.csv')

# dataset = load_json_dataset(Path('../resources/data/Daily-Dialog/DD/train.json'))['test']
# print(dataset)
# pprint(dataset[0])
# dataset = dataset.map(get_src_tgt_question_counts, batched=False, desc="Counting questions")
# print(dataset)
# print(dataset[0])
# write_features_to_file(dataset, '../resources/data/Daily-Dialog/DD/train_src_tgt_question_counts.csv')