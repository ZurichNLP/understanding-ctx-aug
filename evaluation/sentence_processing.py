#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import numpy as np

try:
    from .tokenization import sentencize_texts
except ImportError:
    from tokenization import sentencize_texts

def count_questions(texts: List[str], lang: str = 'en', verbose: bool = False):
    """
    Counts the number of sentences ending with a question mark
    """
    qc = []    
    for text in sentencize_texts(texts, lang=lang, verbose=verbose):
        text_qu_cnt = 0
        for sentence in text:
            if sentence.endswith('?'):
                text_qu_cnt += 1
        qc.append(text_qu_cnt)
    return np.array(qc)

def count_exclamations(texts: List[str], lang: str = 'en'):
    """
    Counts the number of sentences ending with an exclamation mark
    """
    ec = []    
    for text in sentencize_texts(texts):
        text_ex_cnt = 0
        for sentence in text:
            if sentence.endswith('!'):
                text_ex_cnt += 1
        ec.append(text_ex_cnt)
    return np.array(ec)