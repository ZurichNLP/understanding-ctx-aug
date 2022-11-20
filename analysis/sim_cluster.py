#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/agglomerative.py

Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.

Usgage

"""

from argparse import ArgumentParser
from typing import List, Tuple, Dict, Set, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def parse_args():
    ap = ArgumentParser()
    ap.add_argument('-m', '--model', type=str, required=False, default='all-MiniLM-L6-v2')
    ap.add_argument('-t', '--threshold', type=float, required=False, default=0.5)
    ap.add_argument('-i', '--infile', type=str, required=False, default=None)
    ap.add_argument('-v', '--verbose', action='store_true')
    return ap.parse_args()
    
def read_lines(infile):
    lines = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def cluster(corpus: List[str], model: str = 'all-MiniLM-L6-v2', threshold: str = 0.5, verbose: bool = False):
    
    # Load the model
    embedder = SentenceTransformer(model)
    
    # Encode corpus
    corpus_embeddings = embedder.encode(corpus)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])


    # print("Clusters:", len(clustered_sentences))
    if verbose:
        # breakpoint()
        for k in sorted(clustered_sentences, key=lambda k: len(clustered_sentences[k]), reverse=False):
            print("Cluster ", k)
            print("\t", "\t".join(clustered_sentences[k]))
            print("")
        # largest_clusters = [k for k in sorted(clustered_sentences, key=lambda k: len(clustered_sentences[k]), reverse=True)]
        # print(f"Largest clusters")
        # for cluster_id in largest_clusters[:10]:
            # print(f"Cluster {cluster_id}: {clustered_sentences[cluster_id]}")
        
        # for i, cluster in clustered_sentences.items():
        #     print("Cluster ", i+1)
        #     print(cluster)
        #     print("")

    return clustered_sentences

if __name__ == '__main__':
    
    args = parse_args()
    
    # Corpus with example sentences
    if args.infile:
        corpus = read_lines(args.infile)
    else:
        # python analysis/sim_cluster.py -m paraphrase-MiniLM-L6-v2 -t 1.2 works well
        # corpus = [
        #     'A man is eating food.', # 0
        #     'A man is eating a piece of bread.',# 0
        #     'A man is eating pasta.', # 0
        #     'The girl is carrying a baby.', # 1
        #     'The baby is carried by the woman', # 1
        #     'A man is riding a horse.', # 2
        #     'A man is riding a white horse on an enclosed ground.', # 2
        #     'A monkey is playing drums.', # 3
        #     'Someone in a gorilla costume is playing a set of drums.', # 3
        #     'A cheetah is running behind its prey.', # 4
        #     'A cheetah chases prey on across a field.', # 4
        #     'A man is making something to eat.', # 5
        #     'Food is being prepared.' # 5
        #     ]

        # python analysis/sim_cluster.py -m paraphrase-MiniLM-L6-v2 -t 0.7 works well
        corpus = [
            'Do you?',
            'Don\'t you?',
            'Don\'t you?!',
            'Don\'t you!?',
            'Don\'t you!????',
            'Why?',
            'Why!?',
            'Why not?',
            "That's pretty cool.",
            "That's pretty cool!",
            "That's pretty cool?",
        ]

    print(f'Unique sentences: {len(set(corpus))}\tTotal sentences: {len(corpus)}\tRatio: {len(set(corpus)) / len(corpus)}')
    
    clustered_sentences = cluster(corpus, args.model, args.threshold, args.verbose)
    print(f'Unique clusters: {len(clustered_sentences)}\tTotal sentences: {len(corpus)}\tRatio: {len(clustered_sentences) / len(corpus)}')
    
