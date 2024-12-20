#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# sentence-level (tokenized)
CUDA_VISIBLE_DEVICES=0 python extract_and_embed_sentences_from_dataset.py ../resources/data/Topical-Chat/KGD -t -s train valid_freq test_freq
CUDA_VISIBLE_DEVICES=0 python extract_and_embed_sentences_from_dataset.py ../resources/data/Commonsense-Dialogues/CSD -t
CUDA_VISIBLE_DEVICES=0 python extract_and_embed_sentences_from_dataset.py ../resources/data/Daily-Dialog/DD -t

# turn-level
CUDA_VISIBLE_DEVICES=0 python extract_and_embed_sentences_from_dataset.py ../resources/data/Topical-Chat/KGD -s train valid_freq test_freq
CUDA_VISIBLE_DEVICES=0 python extract_and_embed_sentences_from_dataset.py ../resources/data/Daily-Dialog/DD
CUDA_VISIBLE_DEVICES=0 python extract_and_embed_sentences_from_dataset.py ../resources/data/Commonsense-Dialogues/CSD

"""

import sys
import json
from pathlib import Path
from typing import List, Tuple
import re
import argparse
    
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from mosestokenizer import MosesTokenizer, MosesSentenceSplitter
from sentence_transformers import SentenceTransformer # type: ignore

splitsents = MosesSentenceSplitter('en')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Input file containing dialogues")
    parser.add_argument("-t", "--tokenize_sents", action="store_true", help="Split sentences")
    parser.add_argument("-s", "--splits", nargs='+', default=['train', 'valid', 'test'], help="Dataset splits")
    parser.add_argument("-m", "--model", default="all-mpnet-base-v2", help="Sentence Transformer model")
    return parser.parse_args()

def sent_tokenize(text: str, lang: str = 'en') -> List[str]:
    """
    Split a 'text' at sentence boundaries.
    """
    return splitsents([text])

def gather_sentences(infile, tokenize_sents: bool = False):
    """
    Expects jsonl file with keys 'turns' List[str] and 'target' (str)
    """
    sentences = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            for turn in data['turns']:
                turn = re.sub('(<speaker1>|<speaker2>)', '', turn).strip()
                if turn:
                    if tokenize_sents:
                        sentences.extend(sent_tokenize(turn))
                    else:
                        sentences.append(turn)
            target = re.sub('(<speaker1>|<speaker2>)', '', data['target']).strip()
            if tokenize_sents:
                sentences.extend(sent_tokenize(target))
            else:
                sentences.append(target)
    return sentences

def encode_sentences(sentences: List[str], model) -> Tuple[List[str], List[List[float]]]:
    """
    Encode 'sentences' using the Sentence Transformer model.
    """
    embeddings = model.encode(sentences)
    assert len(sentences) == len(embeddings)
    return sentences, embeddings

def process_dataset(data_dir, model_name: str, tokenize_sents: bool = False, splits=['train', 'valid', 'test']):
    """

    """
    model = SentenceTransformer(model_name)
    
    for split in splits:
        infile = Path(data_dir) / f"{split}.json"
        print(f"Processing {infile}")
        if not infile.exists():
            raise FileNotFoundError(f"File {infile} not found.")

        sentences = gather_sentences(infile, tokenize_sents=tokenize_sents)
        print(f"Found {len(sentences)} sentences.")

        sentences, embeddings = encode_sentences(sentences, model)
        
        # write the sentences to a file
        if tokenize_sents:
            outfile = Path(infile).with_suffix('.sents.txt')
        else:
            outfile = Path(infile).with_suffix('.turns.txt')

        with open(outfile, 'w', encoding='utf8') as f:
            for i, sent in enumerate(sentences, 1):
                f.write(sent + '\n')
        print(f"Written {i} sentences to {outfile}")

        # write the embeddings to a file
        emb_file = Path(outfile).with_suffix(f'.{model_name}.npy')
        np.save(emb_file, embeddings)
        print(f"Written embeddings to {emb_file}")

    return

if __name__ == "__main__":

    args = parse_args()
    print(args)
    process_dataset(args.infile, tokenize_sents=args.tokenize_sents, model_name=args.model, splits=args.splits)

    print("Done.")
