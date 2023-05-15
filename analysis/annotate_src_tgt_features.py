#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

NBI: output file will be infered from the input file name and the feature type
NBII: run on GPU for sentiment features

python -m analysis.annotate_src_tgt_features \
    resources/data/Topical-Chat/KGD/train.json \
    question

python -m analysis.annotate_src_tgt_features \
    resources/data/Topical-Chat/KGD/train.json \
    question

python -m analysis.annotate_src_tgt_features \
    resources/data/Commonsense-Dialogues/CD/train.json \
    question

python -m analysis.annotate_src_tgt_features \
    resources/data/DailyDialog/DD/train.json \
    question

"""

import sys
from pathlib import Path
# base_dir = str(Path(sys.path[0]) / "..")
# sys.path.insert(0, base_dir)
# print(sys.path)

import json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from collections import Counter
from typing import List
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from data import preprocess_topical_chat_dataset, preprocess_function, load_data, prepare_data_for_model
from evaluation.sentiment import classify_sentiment_with_vader, parse_vader_result, classify_sentiment

from analysis.sim_metrics import simple_matching_coefficient, compute_sim

def load_json_dataset(dataset: Path):   
    split = dataset.stem
    extension = dataset.suffix.strip('.')
    print(split, extension, str(dataset))
    dataset_dict = load_dataset(extension, data_files={'test': str(dataset)})
    return dataset_dict # concatenate_datasets(loaded_datasets)

def flatten(l: List[List]) -> List:
    """flattens a list of lists"""
    return [item for sublist in l for item in sublist]

def classify_turns(turns):
    flattened_turns = flatten(turns)
    flattened_preds = classify_sentiment(flattened_turns, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=256)
    flattened_preds = np.array([1 if x['label'] == 'POSITIVE' else 0 for x in flattened_preds])
    preds = flattened_preds.reshape(len(turns), len(turns[0]))
    print(preds.shape)
    return preds

def classify_tgts(tgts):
    preds = classify_sentiment(tgts, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=256)
    preds = np.array([1 if x['label'] == 'POSITIVE' else 0 for x in preds])
    return preds

if __name__ == '__main__':
    
    dataset = sys.argv[1] # resources/data/Topical-Chat/KGD/test_freq.json
    feature_type = sys.argv[2] # sentiment

    # infer outfile name
    outpath = Path(dataset).parent / 'analysis' 
    outpath.mkdir(exist_ok=True, parents=True)
    outfile = outpath / f'{Path(dataset).stem}_{feature_type}_features.csv'

    print(f'inferred outfile: {outfile}')

    dataset = load_json_dataset(Path(dataset))
    # print(dataset)

    # src_texts = [[k] + turns for k, turns in zip(dataset['test']['knowledge'], dataset['test']['turns'])]
    src_texts = [turns for turns in dataset['test']['turns']]
    tgt_texts = dataset['test']['target']

    if feature_type == 'sentiment': # run on GPU!
        src_preds = classify_turns(src_texts)
        tgt_preds = classify_tgts(tgt_texts)
    elif feature_type == 'question':
        src_preds = np.array([[1 if '?' in x else 0 for x in turn] for turn in src_texts])
        tgt_preds = np.array([1 if '?' in x else 0 for x in tgt_texts])
    else:
        raise NotImplementedError

    # to to disk so we can load it later
    df = pd.DataFrame({'tgt_texts': tgt_texts, f'tgt_{feature_type}': tgt_preds})
    for i in range(len(src_texts[0])):
        df[f'src_{i+1}'] = [s[i] for s in src_texts] # add knowledge text
        df[f'src_{i+1}_{feature_type}'] = src_preds[:,i]
    df.to_csv(outfile, index=False)
    print(f'wrote {feature_type} features DF to {outfile}')