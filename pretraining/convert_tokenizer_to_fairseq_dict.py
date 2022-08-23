#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given a Hugging Face Tokenizer, this script loads the tokenizer's vocab.json file and produces dict.txt file requried for fairseq

output might look sth like this:

    Ġ. 1234
    Ġ, 1234
    Ġoch 1234
    Ġi 1234
    Ġatt 1234
    ĠÃ¤r 1234
    Ġsom 1234
    ...

where 1234 is the dummy frequency occurrence of the token.

Example usage:

    python convert_tokenizer_to_fairseq_dict.py -v resources/bookcorpus/tokenizer/vocab.json -o resources/bookcorpus/bin

"""

import json
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--vocab", required=False, type=str, help="Tokenizer's vocab.json file")
ap.add_argument("-o", "--out_dir", required=False, type=str, help="Output directory")
args = ap.parse_args()

outfile = Path(args.out_dir) / "dict.txt"

with open(args.vocab, 'r') as f:
    vocab = json.load(f)

with open(outfile, 'w', encoding='utf8') as f:
    c = 0
    for token in vocab:
        if token not in ["<s>","<pad>", "</s>", "<unk>"]: # these tokens are assumed to be present by fairseq and not needed in the dict
            c += 1
            f.write(f'{token} {str(1234)}\n')

print(f'Wrote {c} tokens to {outfile}')