#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example Usage:
    python collect_contexts.py \
        --corpus_file resources/data/Topical-Chat/KGD/train.json \
        --outfile resources/data/Topical-Chat/KGD/contexts/train_questions.txt --extract q

    python collect_contexts.py \
        --corpus_file resources/data/Topical-Chat/KGD/train.json \
        --outfile resources/data/Topical-Chat/KGD/contexts/train_exclamations.txt --extract e
"""

import argparse
from pathlib import Path
import re

import spacy
from datasets import load_dataset
from tqdm import tqdm


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus_file', type=str, default='resources/data/Topical-Chat/KGD/train.json', help='dataset name')
    ap.add_argument('--outfile', type=str, required=True, help='output file name')
    ap.add_argument('--max_contexts', type=int, default=None, help='max number of contexts')
    ap.add_argument('--extract', type=str, default='q', choices=['q','e'], help='q for questions, e for exclamations')
    return ap.parse_args()

def load_corpus(corpus_file):
    if 'topical-chat' in corpus_file.lower():
        extension = corpus_file.split(".")[-1]
        dataset_dict = load_dataset(extension, data_files=corpus_file)
        corpus_sents = dataset_dict['train']['target']
    elif 'commonsense' in corpus_file.lower():
        dataset_dict = load_dataset('json', data_files=corpus_file)
        corpus_sents = dataset_dict['train']['target']
        # note: for commonsense, we also consider sentences from the context
        for turns in dataset_dict['train']['turns']:
            corpus_sents.extend(turn for turn in turns if turn != '')
    elif 'dailydialog' in corpus_file.lower():
        dataset_dict = load_dataset('json', data_files=corpus_file)
        corpus_sents = dataset_dict['train']['target']

    print(f'Corpus sentences: {len(corpus_sents)}')

    return corpus_sents

def clean(string):
    """remove speaker tags"""
    return re.sub(r'<speaker\d>\s?', '', string).strip()

def extract_questions(corpus, nlp, sentence_length_threshold=6):
    """extract questions from corpus"""
    questions = set()
    for doc in tqdm(nlp.pipe(corpus, batch_size=100, n_process=8), total=len(corpus)):
        for sent in doc.sents:
            if len(sent) >= sentence_length_threshold and sent.text.strip().endswith('?'):
                questions.add(clean(sent.text))
    print(f'Found {len(questions)} questions in corpus')
    return questions

def extract_exclamations(corpus, nlp, sentence_length_threshold=6):
    """extract exlamations from corpus"""
    exlamations = set()
    for doc in tqdm(nlp.pipe(corpus, batch_size=100, n_process=8), total=len(corpus)):
        for sent in doc.sents:
            if len(sent) >= sentence_length_threshold and sent.text.strip().endswith('!'):
                exlamations.add(clean(sent.text))
    print(f'Found {len(exlamations)} exlamations in corpus')
    return exlamations

def write_to_outfile(iterable, outfile, total=None):

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    
    with open(outfile, 'w', encoding='utf8') as f:
        for i, item in enumerate(iterable):
            if total is not None and i > total:
                break
            f.write(item + '\n')
    
    print(f'Wrote {i+1} items to {outfile}')

    return

def main(args):

    # Load dataset
    corpus = load_corpus(args.corpus_file)
    # Load spacy model
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("sentencizer")
    
    if args.extract == 'q': # Extract questions
        questions = extract_questions(corpus, nlp)
        write_to_outfile(questions, args.outfile, args.max_contexts)
    elif args.extract == 'e': # Extract exclamations
        exlamations = extract_exclamations(corpus, nlp)
        write_to_outfile(exlamations, args.outfile, args.max_contexts)

if __name__ == '__main__':
    args = set_args()
    main(args)