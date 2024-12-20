#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python compute_tsne_reduction.py ../resources/data/Commonsense-Dialogues/CSD/test.turns.all-mpnet-base-v2.npy ../resources/data/Commonsense-Dialogues/CSD/valid.turns.all-mpnet-base-v2.npy ../resources/data/Commonsense-Dialogues/CSD/train.turns.all-mpnet-base-v2.npy
python compute_tsne_reduction.py ../resources/data/Daily-Dialog/DD/test.turns.all-mpnet-base-v2.npy ../resources/data/Daily-Dialog/DD/valid.turns.all-mpnet-base-v2.npy ../resources/data/Daily-Dialog/DD/train.turns.all-mpnet-base-v2.npy
python compute_tsne_reduction.py ../resources/data/Topical-Chat/KGD/test_freq.turns.all-mpnet-base-v2.npy ../resources/data/Topical-Chat/KGD/valid_freq.turns.all-mpnet-base-v2.npy ../resources/data/Topical-Chat/KGD/train.turns.all-mpnet-base-v2.npy
"""


import sys
import json
from pathlib import Path
from typing import List, Tuple
import re
import numpy as np

# avoid OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from sklearn.manifold import TSNE

def load_sentences(file_path: Path):
    with open(file_path, 'r', encoding='utf8') as f:
        sentences = [line.strip() for line in f]
    return sentences

def load_embeddings(file_path: Path):
    return np.load(file_path)    

if __name__ == '__main__':

    embeddings_files = sys.argv[1:]

    for embeddings_file in embeddings_files:
        if not Path(embeddings_file).exists():
            print(f'File {embeddings_file} does not exist')
            sys.exit(1)
        
        else:
            embeddings = load_embeddings(embeddings_file)

            print(f'Length of embeddings: {len(embeddings)}')

            tsne_model = TSNE(n_components=2) #n_components means the lower dimension
            print('Fitting t-SNE model ...')
            low_dim_data = tsne_model.fit_transform(embeddings)
            
            assert low_dim_data.shape[0] == embeddings.shape[0]
            print('Lower dim data has shape', low_dim_data.shape)

            tsne_file = Path(embeddings_file).with_suffix('.tsne.npy')
            np.save(tsne_file, low_dim_data)

            print(f'Saved t-SNE reduction to {tsne_file}')