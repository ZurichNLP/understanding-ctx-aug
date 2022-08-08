#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import spacy
from datasets import load_dataset
from tqdm import tqdm


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus_file', type=str, default='data/Topical-Chat/KGD/train.json', help='dataset name')
    ap.add_argument('--outfile', type=str, required=True, help='output file name')
    ap.add_argument('--max_contexts', type=int, default=None, help='max number of contexts')
    return ap.parse_args()

def load_corpus(corpus_file):
    if 'Topical-Chat' in corpus_file:
        extension = corpus_file.split(".")[-1]
        dataset_dict = load_dataset(extension, data_files=corpus_file)
        corpus_sents = dataset_dict['train']['target']
    return corpus_sents

def extract_questions(corpus, nlp, sentence_length_threshold=6):
    questions = set()
    for doc in tqdm(nlp.pipe(corpus, batch_size=100, n_process=8), total=len(corpus)):
        for sent in doc.sents:
            if len(sent) >= sentence_length_threshold and sent.text.strip().endswith('?'):
                questions.add(sent.text)
    print(f'Found {len(questions)} questions in corpus.')
    return questions

def write_to_outfile(iterable, outfile, total=None):
    
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    
    with open(outfile, 'w', encoding='utf8') as f:
        for i, item in enumerate(iterable):
            if total is not None and i > total:
                break
            f.write(item + '\n')
    
    print(f'Wrote {i+1} items to {outfile}.')

    return

def main(args):

    # Load dataset
    corpus = load_corpus(args.corpus_file)
    # Load spacy model
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("sentencizer")
    # Extract questions
    questions = extract_questions(corpus, nlp)    
    # Write to outfile
    write_to_outfile(questions, args.outfile, args.max_contexts)


if __name__ == '__main__':
    args = set_args()
    main(args)