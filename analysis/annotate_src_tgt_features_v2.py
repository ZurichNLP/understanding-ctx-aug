import argparse
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from multiprocessing import Pool
from evaluation.sentence_processing import count_questions
from mosestokenizer import MosesSentenceSplitter, MosesTokenizer

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example usage:

    python annotate_src_tgt_features.py \
        --dataset "resources/data/Topical-Chat/KGD/test_freq.json" \
        --output_file "resources/data/Topical-Chat/KGD/kgd_test_freq_src_tgt_question_counts.csv"

"""



# set args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

sent_tokenizer = MosesSentenceSplitter('en')

def load_lines(file_path):   
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            yield line['turns'] + [line['target']]


def get_sentences(lines):
    print(len(lines))
    sents = [[sent_tokenizer(line)] for line in lines]
    return sents


def get_src_tgt_question_counts(line):
    result = count_questions(line, verbose=False).tolist()
    return result

def get_src_tgt_question_counts_batch(batch):
    results = []
    for line in batch:
        result = count_questions(line, verbose=False).tolist()
        results.append(result)
    return results

# write to csv
def write_results_to_file(results, outfile):
    
    with open(outfile, 'w', encoding='utf8') as f:
        for i, item in results:
            f.write(f"{item}\n")
            if i < 5:
                print(item)
                
    print(f'Wrote {i+0} items to {outfile}')

    return

if __name__ == '__main__':
    args = parser.parse_args()

    lines = [l for l in load_lines(Path(args.dataset))]
    print(f'Loaded {len(lines)} examples from {args.dataset}')
    print('Example:')
    print(lines[0])

    for line in lines[:10]:
        # sentence tokenize
        sentences = get_sentences(line)
        print(sentences)

    # # Split the dataset into chunks for multiprocessing
    # num_processes = 1  # Number of processes to use
    
    # # Split the dataset into chunks for multiprocessing
    # chunk_size = len(lines) // num_processes
    # chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    # # breakpoint()
    # # Process the chunks in parallel
    # with Pool(num_processes) as pool:
    #     results = list(tqdm(pool.imap(get_src_tgt_question_counts, chunks), total=len(chunks)))
    # # results = [get_src_tgt_question_counts(chunk) for chunk in chunks]

    # # Flatten the results
    # results = [(i, item) for i, chunk in enumerate(results) for item in chunk]

    # # Write the results to a CSV file
    # write_results_to_file(results, args.output_file)