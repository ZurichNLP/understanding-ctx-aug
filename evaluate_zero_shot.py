#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import spacy
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('generations_file', type=str, help='generated outputs from inference.py')
    return ap.parse_args()

def read_lines(file, sep='\t'):    
    lines = []
    with open(file, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            line = line.split(sep)[0]
            lines.append(line.strip())
    return lines

def count_questions(corpus, nlp, sentence_length_threshold=6):
    questions = []
    for doc in tqdm(nlp.pipe(corpus, batch_size=100, n_process=8), total=len(corpus)):
        doc_qu_cnt = 0
        for sent in doc.sents:
            if sent.text.strip().endswith('?'):
                doc_qu_cnt += 1
        questions.append(doc_qu_cnt)
    return np.array(questions)

# def write_to_outfile(iterable, outfile, total=None):
    
#     Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    
#     with open(outfile, 'w', encoding='utf8') as f:
#         for i, item in enumerate(iterable):
#             if total is not None and i > total:
#                 break
#             f.write(item + '\n')
    
#     print(f'Wrote {i+1} items to {outfile}.')

#     return

def main(args):

    # Load dataset
    generated_outputs = read_lines(args.generations_file)
    
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("sentencizer")
    
    # Extract questions
    questions = count_questions(generated_outputs, nlp)    
    
    df = pd.DataFrame.from_dict(
        {'outputs': generated_outputs,
        'question_counts': questions
        }
    )

    print('Outputs:')
    print(df['outputs'].describe())
    print('Questions:')
    print(df['question_counts'].describe())
    # df.to_csv('data/Topical-Chat/KGD/train_qu_cnt.csv', index=False)
    # write_to_outfile(questions, args.outfile, args.max_contexts)


if __name__ == '__main__':
    args = set_args()
    main(args)