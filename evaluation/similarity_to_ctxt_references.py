#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script can be used to compute the similarity of a model's predictions to a set of questions used as context phrases for single_qu_ctxt_aug5 experiments.

Usage:
    
    python evaluation/similarity_to_ctxt_references.py seed_ctxt_qus

    python evaluation/similarity_to_ctxt_references.py seed_ctxt_pos

"""

import sys
import numpy as np
import pandas as pd

from reference_metrics import compute_bleu, compute_rouge, compute_exact_match, compute_bertscore
from evaluation import read_lines

# these are the questions randomly sampled from train_questions.txt for each generation seed
ctxts = {
    'seed_ctxt_qus': {
        0: 'I wonder if the brain processes Braille different than regular reading?',
        42: 'Did you know that the title refers to the entire clan and not just his character?',
        983: 'Would you like to discuss about a different topic, such as the history of telephone, etc. ?',
        8630: 'Do you shop at walmart at all?',
        284: 'So are you a Jordan guy?',
        },
    'seed_ctxt_pos': {
        0: "It's great to",
        42: "That's awesome",
        983: "It's wonderful to",
        8630: "It's wonderful to",
        284: "That's awesome",
    }
}

# https://stackoverflow.com/a/66789625/4649965
def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res

def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df

def average_bertscore(d):
    """
    Bertscore returns a list of precision, recall, and F1 scores for each prediction/reference pair.
    We average the scores across all prediction/reference pairs.
    """
    new_d = {}
    for k in d.keys():
        v = d[k]
        if isinstance(v, list) and len(v) > 1:
            new_d[f'bertscore_{k}_mean'] = np.mean(v)
            new_d[f'bertscore_{k}_std'] = np.std(v)
        else:
            new_d[f'bertscore_{k}'] = v
    return new_d

if __name__ == '__main__':

    ctxt_references = sys.argv[1] # 'seed_ctxt_qus' or 'seed_ctxt_pos'
    
    if ctxt_references not in ctxts:
        raise ValueError(f'Invalid context references: {ctxt_references}')
    elif ctxt_references == 'seed_ctxt_qus':
        ctxts = ctxts['seed_ctxt_qus']
        file_suffix = '_ctxt=5-train_questions-1.txt'
    elif ctxt_references == 'seed_ctxt_pos':
        ctxts = ctxts['seed_ctxt_pos']
        file_suffix = '_ctxt=5-pos_sents-1.txt'
        
    outfile = f'resources/{ctxt_references}_similarity_scores.csv'
    
    results = {}
    dfs = []
    for model_seed in [23, 42, 1984]:
        results[model_seed] = {}
        for model in ['bart_small-MLM_PS', 'bart_small-MLM', 'bart_small-PS', 'bart_small-SI_mass', 'bart_small-SI_t5', 'bart_small-SI_bart', 'bart_small_rndm', 'bart_base', 't5_small', 't5_lm_small', 't5v11_small']:
            results[model_seed][model] = {}
            for seed in ctxts:
                
                file = f'resources/models/seed_{model_seed}/ft/{model}/outputs/generations_test_freq_seed={seed}_ml=40_lp=1.0_ns=1_bs=4_ds=1_temp=0.7_tk=0_tp=0.9{file_suffix}'
                print(file)
                
                predictions = read_lines(file)
                references = [[ctxts[seed]]] * len(predictions)
                bleu = compute_bleu(predictions, references, verbose=True)
                rouge = compute_rouge(predictions, references, verbose=True)
                exact = compute_exact_match(predictions, [' '.join(r) for r in references], verbose=True)
                bert = average_bertscore(compute_bertscore(predictions, [' '.join(r) for r in references], verbose=True))
                
                results[model_seed][model][seed] = {**rouge, **exact, **bert}
                results[model_seed][model][seed]['bleu'] = bleu['score']
                
                simdf = nested_dict_to_df(results)
                print(f'saved {simdf.shape} results to {outfile}')
                simdf.to_csv(outfile)