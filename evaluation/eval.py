#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Optional, Union
import json
import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

try:
    from .perplexity import score_ppl
    from .sentence_processing import count_questions
    from .reference_metrics import compute_rouge, compute_bleu, compute_meteor
    from .distinct import distinct_n
    from .tokenization import tokenize_texts
except ImportError:
    from perplexity import score_ppl
    from sentence_processing import count_questions
    from reference_metrics import compute_rouge, compute_bleu, compute_meteor
    from distinct import distinct_n
    from tokenization import tokenize_texts


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('generations', type=str, help='file or diectory of files containing generated outputs from inference.py')
    ap.add_argument('-r', '--references_file', type=str, default=None, help='e.g. `resources/data/Topical-Chat/KGD/test_freq.json`')
    ap.add_argument('-o', '--outfile', type=str, default=None, help='')
    ap.add_argument('--output_dir', type=str, default=None, help='')

    return ap.parse_args()

def read_lines(file: str, sep: str = '\t'):    
    lines = []
    with open(file, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            line = line.split(sep)[0]
            lines.append(line.strip())
    return lines

def reshape_data(data: List[Dict]):
    """
    
    """
    reshaped = {}
    keys = list(data[0].keys())
    for key in keys:
        reshaped[key] = []
        for line in data:
            reshaped[key].append(line[key])
    return reshaped

def read_json_lines(file: str):
    lines = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(json.loads(line.strip()))
    return reshape_data(lines)
    
def uniq_response_ratio(texts: List[str]):
    return len(set(texts)) / len(texts)


def compute_reference_free_metrics(
    sys_outputs: List[str], 
    verbose: bool = False
    ):
    """
    reference-free metrics
    """

    results = {}

    results['uniq'] = uniq_response_ratio(sys_outputs)
    
    qc = count_questions(sys_outputs)  
    results['qc_turn_level'] = sum([1 for i in qc if i > 0]) / len(qc)
    results['qc_sent_level'] = qc.sum() / len(qc)
    
    ppl_mean, ppl_std = score_ppl(sys_outputs, batch_size=128)
    results['ppl_mean'] = ppl_mean
    results['ppl_std'] = ppl_std

    dist = distinct_n(tokenize_texts(sys_outputs))
    results.update(dist)

    return results

def compute_reference_based_metrics(
    sys_outputs: List[str], 
    references: List[List[str]],
    tag: str = '',
    verbose: bool = False
    ):

    """
    reference-based metrics (BLEU, ROUGE, METEOR) for KGD

    :tag: 't' for target, 'k' for knowledge, 'd' for dialog
    """
    results = {}

    bleu = compute_bleu(sys_outputs, references, is_tokenized=False)
    rouge = compute_rouge(sys_outputs, references, is_tokenized=False)
    meteor = compute_meteor(sys_outputs, references, is_tokenized=False)
    
    if tag:
        tag = '_' + tag

    results[f'bleu{tag}'] = bleu['score'] if bleu is not None else None
    results[f'rouge1{tag}'] = rouge['rouge1'] if rouge is not None else None
    results[f'meteor{tag}'] = meteor['meteor'] if meteor is not None else None
    
    return results    

def validate_system_outputs(sys_outputs: List[str]):
    """
    check if system outputs are valid
    """
    problematic = []
    for i in range(len(sys_outputs)):
        if len(sys_outputs[i].strip().split()) <= 1:
            sys_outputs[i] = 'n/a.'
            problematic.append(i)
    if len(problematic) > 0:
        print(f'[!] {len(problematic)} problematic system outputs: Check the following lines: {problematic}')
    return sys_outputs    

def score_kgd_generation(
    sys_outputs: List[str], 
    targets: Optional[List[str]],
    knowledge_snippets: Optional[List[str]],
    dialogs: Optional[List[str]],
    verbose: bool = False
    ):

    results = {}

    validate_system_outputs(sys_outputs)

    results.update(compute_reference_free_metrics(sys_outputs, verbose=verbose))
    
    if targets is not None:
        results.update(compute_reference_based_metrics(sys_outputs, targets, 't', verbose))
    if knowledge_snippets is not None:
        results.update(compute_reference_based_metrics(sys_outputs, knowledge_snippets, 'k', verbose))
    if dialogs is not None:
        results.update(compute_reference_based_metrics(sys_outputs, dialogs, 'd', verbose))

    print(f'Scored {len(sys_outputs)} system outputs')

    return results


def write_to_csv(results: Dict, outfile: Optional[str]):

    df = pd.DataFrame([results])
    if not outfile:
        print(df.to_string())
    else:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outfile, index=False)
    return

def main(args):

    if Path(args.generations).is_dir(): # run evaluation for each file in the directory
        results = []
        for generations_file in Path(args.generations).glob('*.txt'):
            print(f'Processing {generations_file}')
            sys_outputs = read_lines(generations_file)
            source = read_json_lines(args.references_file) if args.references_file is not None else None

            refs_t = [[i] for i in source['target']]
            refs_k = [[i] for i in source['knowledge']]
            refs_d = [[' '.join(i)] for i in source['turns']]

            result = score_kgd_generation(
                sys_outputs=sys_outputs,
                targets=refs_t,
                knowledge_snippets=refs_k,
                dialogs=refs_d,
            )
            result['file'] = generations_file.name
            results.append(result)
        df = pd.DataFrame(results)    
        df.to_csv(outfile, index=False)

    elif Path(args.generations).is_file():

        sys_outputs = read_lines(args.generations)
        source = read_json_lines(args.references_file) if args.references_file is not None else None
        
        refs_t = [[i] for i in source['target']]
        refs_k = [[i] for i in source['knowledge']]
        refs_d = [[' '.join(i)] for i in source['turns']]

        results = score_kgd_generation(
            sys_outputs=sys_outputs,
            targets=refs_t,
            knowledge_snippets=refs_k,
            dialogs=refs_d,
            )

        results['file'] = args.generations_file
        
        write_to_csv(results, args.outfile)

if __name__ == '__main__':
    args = set_args()
    main(args)