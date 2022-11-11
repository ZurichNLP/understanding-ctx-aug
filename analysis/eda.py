#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loosely based on https://towardsdatascience.com/exploratory-text-analysis-in-python-8cf42b758d9e
"""

import sys
import re
import random
from argparse import ArgumentParser
from typing import List, Dict, Optional, Union
from collections import Counter
import string

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk import FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter

from scipy import stats

import plotext as plt # https://github.com/piccolomo/plotext


def parse_args() -> ArgumentParser:
    ap = ArgumentParser()
    ap.add_argument("-i", "--infile", required=True, help="Input file")
    ap.add_argument("-o", "--outfile", required=False, help="Output file")
    ap.add_argument("-l", "--lang", required=False, default='english', help="Language")
    ap.add_argument("-v", "--verbose", type=int, required=False, default=0, help="Verbosity level")
    ap.add_argument("-s", "--show_n", type=int, required=False, default=10, help="Number of examples to show")
    return ap.parse_args()

def read_lines(infile: str) -> List[str]:
    lines = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
    return lines

def word_tokenize(texts: List[str], lang: str = 'english') -> List[List[str]]:
    """
    Tokenize list of string texts
    """
    with MosesTokenizer(lang[:2]) as tokenize:
        if isinstance(texts[0], list):
            tok_texts = []
            for text in tqdm(texts, total=len(texts), desc="Tokenizing Words", disable=True):
                tok_sents = []
                for sent in text:
                    tok_sents.append(' '.join(tokenize(sent)))
                tok_texts.append(tok_sents)
            return tok_texts
        else:
            return [' '.join(tokenize(text)) for text in tqdm(texts, total=len(texts), desc="Tokenizing Words")]

def sent_tokenize(texts: List[str], lang: str = 'english') -> List[List[str]]:
    """
    Split 'texts' at sentence boundaries.
    """
    with MosesSentenceSplitter(lang[:2]) as splitsents:
        return [splitsents([text]) for text in tqdm(texts, total=len(texts), desc="Tokenizing Sentences", disable=True)]

def summarise_ttr(d: Counter, types: str, show_n: int = 10, verbose: bool = False) -> None:
    total = sum(d.values())
    unique = len(d)
    
    if verbose > 1:
        print(f"Most common {types.upper()}:\n\t{d.most_common(show_n)}")
    
    return {
        types: {
            'total': total, 
            'unique': unique,
            'ttr': (unique/total)*100
            }
        }

def summarise_dist(array: np.array):
    """Summarise a distribution."""
    print(stats.describe(array))
    print(stats.mode(array, axis=None, keepdims=True))
    return

def get_doc_stats(texts: List[str], lang: str = 'english', show_n: int = 5, verbose: bool = False) -> None:
    """
    Get doc-level statistics for a list of texts

    * How many unique docs are there?
    * What are the most common docs?
    * What do the longest/shortest docs look like?
    """
    if verbose > 0:
        print(f'\n**** Doc-level stats ****')
    # count unique docs
    doc_counter = Counter(texts)

    # strings vs unique strings (TTR)
    doc_stats = summarise_ttr(doc_counter, 'documents', verbose=verbose)
    
    if verbose > 0:
        # get shortest docs
        shortest_strings = sorted(doc_counter, key=len)[:show_n]
        print(f"Shortest docs:\n\t{shortest_strings}")
        # get longest docs
        longest_strings = sorted(doc_counter, key=len, reverse=True)[:show_n]
        print(f"Longest docs:\n\t{longest_strings}")

    return pd.DataFrame.from_dict(doc_stats, orient='index')

def get_sent_stats(texts: List[str], lang: str = 'english', show_n: int = 10, verbose: bool = False) -> None:
    """
    Get sentence-level statistics for a list of texts

    * How many unique sentences are there?
    * What are the most common sentences?
    * What do the longest/shortest sentences look like?
    * How many sentences are there per document?
    * How many questions / exclamations / no-punct-sents are there?
    """
    if verbose > 0:
        print(f'\n**** Sentence-level stats ****')
    
    sents_per_doc = [len(doc) for doc in texts]

    if verbose > 0:
        print(f'Sentences per document:')
        summarise_dist(np.array(sents_per_doc))

    # # count unique sentences
    sentence_counter = Counter([sentence for doc in texts for sentence in doc])

    # sentences vs unique sentences (TTR)
    sent_stats = summarise_ttr(sentence_counter, 'sentences', verbose=verbose)

    # how many questions are there?
    question_counter = Counter([sentence for doc in texts for sentence in doc if sentence.strip().endswith('?')])
    sent_stats.update(summarise_ttr(question_counter, 'questions'))

    # how many exclamation sentences are there?
    exclamation_counter = Counter([sentence for doc in texts for sentence in doc if sentence.strip().endswith('!')])
    sent_stats.update(summarise_ttr(exclamation_counter, 'exclamations', verbose=verbose))

    # how many sentences have no punctuation?
    no_punct_counter = Counter([sentence for doc in texts for sentence in doc if sentence.strip()[-1] not in ['.', '?', '!']])
    sent_stats.update(summarise_ttr(no_punct_counter, 'no punctuation', verbose=verbose))

    if verbose > 0:
        # get shortest sentences
        shortest_sents = sorted(sentence_counter, key=len)[:show_n]
        print(f"Shortest sentences:\n\t{shortest_sents}")
        # get longest sentences
        longest_sents = sorted(sentence_counter, key=len, reverse=True)[:show_n]
        print(f"Longest sentences:\n\t{longest_sents}")

    return pd.DataFrame.from_dict(sent_stats, orient='index')
    
def get_token_stats(texts: List[str], lang: str = 'english', show_n: int = 25, verbose: bool = False) -> None:
    """
    Get token-level statistics for a list of texts

    * How many ngrams / unique ngram are there?
    * What is the average number of characters per token?
    * What are the most common stop words?
    * What are the most common non-stop unigrams?
    # * What other words occur so often such that could be added to stop words?
    * What are the longest/shortest ngrams?
    """

    if verbose > 0:
        print(f'\n**** Token-level stats ****')

    # tokenize
    # texts = word_tokenize(texts, lang)

    chars_per_token = []
    tokens_per_sent = []
    stop_counter = Counter()
    punct_counter = Counter()
    num_counter = Counter()
    unigram_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()
    fourgram_counter = Counter()
    
    # iterate over texts and collect token stats
    # for text in tqdm(texts, total=len(texts), desc="Collecting token-level stats"):
    # breakpoint()
    for text in texts:
        for sent in text:
            tokens = sent.split()
            chars_per_token.extend([len(token) for token in tokens])
            tokens_per_sent.extend([len(tokens)])
            # stop_counter.update([token for token in tokens if token ])
            stop_counter.update(filter(lambda x: x.lower() in stopwords.words(lang), tokens))
            # punct_counter.update([token for token in tokens if token in string.punctuation])
            punct_counter.update(filter(lambda x: x in string.punctuation, tokens))
            # num_counter.update([token for token in tokens if token.isnumeric()])
            num_counter.update(filter(lambda x: x.isnumeric(), tokens))

            unigram_counter.update(tokens)
            bigram_counter.update(' '.join(ngram) for ngram in ngrams(tokens, 2))
            trigram_counter.update(' '.join(ngram) for ngram in ngrams(tokens, 3))
            fourgram_counter.update(' '.join(ngram) for ngram in ngrams(tokens, 4))
    
    if verbose > 0:
        # get descriptive stats for tokens per sentence
        print(f'Tokens per sentence:')
        summarise_dist(np.array(tokens_per_sent))
        # get descriptive stats for characters per token
        print(f'Characters per token:')
        summarise_dist(np.array(chars_per_token))


    # get number of tokens
    tok_stats = summarise_ttr(unigram_counter, 'tokens', verbose=verbose)
    
    # stop words
    tok_stats.update(summarise_ttr(stop_counter, 'stop words', verbose=verbose))
    
    # punctuation
    tok_stats.update(summarise_ttr(punct_counter, 'punctuation', verbose=verbose))

    # numbers
    tok_stats.update(summarise_ttr(num_counter, 'numbers', verbose=verbose))
    
    # most common non-stop words
    non_stop_unigrams = unigram_counter - stop_counter - punct_counter - num_counter
    tok_stats.update(summarise_ttr(non_stop_unigrams, 'non-stop words', verbose=verbose))
    
    # bigrams
    tok_stats.update(summarise_ttr(bigram_counter, 'bigrams', verbose=verbose))

    # trigrams
    tok_stats.update(summarise_ttr(trigram_counter, 'trigrams', verbose=verbose))

    # fourgrams
    tok_stats.update(summarise_ttr(fourgram_counter, 'fourgrams', verbose=verbose))

    # get shortest tokens
    shortest_tokens = sorted(non_stop_unigrams, key=len)[:show_n*2]
    if verbose > 0:
        print(f"Shortest (non-stop) tokens:\n\t{shortest_tokens}")

    # get longest tokens
    longest_tokens = sorted(non_stop_unigrams, key=len, reverse=True)[:show_n]
    if verbose > 0:
        print(f"Longest (non-stop) tokens:\n\t{longest_tokens}")

    return pd.DataFrame.from_dict(tok_stats, orient='index')

def do_eda(infile: str, lang: str = 'english', verbose: int = 0):
    
    texts = read_lines(infile)

    doc_stats = get_doc_stats(texts, verbose=verbose)
    
    # split into sentences
    texts = sent_tokenize(texts, lang=lang)
    sent_stats = get_sent_stats(texts, verbose=verbose)

    # tokenize    
    texts = word_tokenize(texts, lang=lang)
    tok_stats = get_token_stats(texts, verbose=verbose)
    
    df = pd.concat([doc_stats, sent_stats, tok_stats])
    
    return df


if __name__ == '__main__':
    args = parse_args()
    infile = args.infile
    verbose = args.verbose
    df = do_eda(infile, verbose=verbose)
    print(df)

    plt.simple_stacked_bar(
        df.index.to_list(), 
        [df['unique'].to_numpy(), df['total'].to_numpy()-df['unique'].to_numpy()],
        labels=['uniq', 'total'],  width = 100
        )

    plt.show()