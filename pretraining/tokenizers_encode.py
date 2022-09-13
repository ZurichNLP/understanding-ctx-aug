#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fairseq doesn't do any preprossing of the data, so we can just use 
the tokenize the data on disk before running fariseq-preprocess

This version reads in the raw data as batched lines for memory efficient processing

Example usage:

python tokenizers_encode.py \
    -i resources/bookcorpus/tmp \
    -f train.txt \
    -o resources/bookcorpus/tok \
    --path_to_tokenizer resources/bookcorpus/tokenizer

without mp, processing time should be approx 1.5 hours
406000 sentences in 30 secs:
    ((72782643 / 406000 ) * 30 ) / 60
"""

import time
import argparse
from typing import List, Optional, Dict
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from memory_profiler import profile
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", type=str)
    ap.add_argument("-i", "--in_dir", required=True, type=str, help="Path to directory containing files for tokenizing")
    ap.add_argument("-o", "--output_dir", required=False, type=str, help="Path to the output")
    ap.add_argument("--path_to_tokenizer", required=False, type=str, help="Path to tokenizer to apply")
    ap.add_argument("--as_tokens", action="store_true", help="Output human-readable tokens")
    ap.add_argument("--batch_size", required=False, default=1000, type=int, help="")
    return ap.parse_args()

def tokenize_as_token_ids(examples, tokenizer):
    encodings = tokenizer(examples, truncation=False, return_attention_mask=False, add_special_tokens=False)
    return encodings['input_ids'] # by default, returns dictionaty of 'input_ids' and 'attention_mask' (if avaliable)

def tokenize_as_tokens(examples, tokenizer):
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in examples] # allow batching
    return tokens

def encode(lines, tokenizer):
    token_ids = tokenize_as_token_ids(lines, tokenizer)
    tokens = tokenize_as_tokens(token_ids, tokenizer)
    return list(zip(token_ids, tokens))

def read_batched_lines(file, batch_size=1000):
    """Read lines from file in batches"""
    lines = []
    with open(file, 'r') as f:
        for line in f:
            if len(lines) == batch_size:
                yield lines
                lines = [line.strip()] # reset lines with current line
            else:
                lines.append(line.strip())
    if lines:
        # print(f"{len(lines)} lines remaining")
        yield lines # in case there are some lines left


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_tokenizer)
    tokenizer.model_max_length = tokenizer.model_max_length * 20 # set as large number to avoid warning

    infile = str(Path(args.in_dir) / args.filename)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    inds_outf = str(Path(args.output_dir) / Path(args.filename).with_suffix('.ind'))
    toks_outf = str(Path(args.output_dir) / Path(args.filename).with_suffix('.tok'))

    t0 = time.time()
    line_c = 0
    with open(inds_outf, 'w', encoding='utf8') as ind_f:
        with open(toks_outf, 'w', encoding='utf8') as toks_f:
            for batched_lines in read_batched_lines(infile, args.batch_size):
                encoded_lines = encode(batched_lines, tokenizer)            
                for token_ids, tokens in encoded_lines:
                    ind_f.write(f"{' '.join(map(str, token_ids))}\n") 
                    toks_f.write(f"{' '.join(tokens)}\n")
                line_c += len(encoded_lines)
                if line_c % 100000 == 0:
                    logger.info(f"{line_c} lines processed. Wall time: {time.time() - t0:.4f} secs")
    
    t1 = time.time()
    logger.info(f"*** Tokenized {line_c} sentences from file {infile} in {t1 - t0:.4f} seconds. ***")
    logger.info(f"*** Outfiles: {inds_outf} --- {toks_outf} ***")


if __name__ == "__main__":
    args = set_args()
    main(args)

