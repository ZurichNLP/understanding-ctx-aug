#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import numpy as np

try:
    from .tokenization import sentencize_texts
except ImportError:
    from tokenization import sentencize_texts

def count_questions(texts: List[str], lang: str = 'en'):
    """
    Counts the number of sentences ending with a question mark
    """
    qc = []    
    for text in sentencize_texts(texts):
        text_qu_cnt = 0
        for sentence in text:
            if sentence.endswith('?'):
                text_qu_cnt += 1
        qc.append(text_qu_cnt)
    return np.array(qc)

# import spacy
# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe("sentencizer")

# def count_questions_spacy():
#     qc = []
#     for doc in tqdm(nlp.pipe(texts, batch_size=100, n_process=8), desc="Splitting Sentences", total=len(texts)):
#         doc_qu_cnt = 0
#         for sent in doc.sents:
#             if sent.text.strip().endswith('?'):
#                 doc_qu_cnt += 1
#         qc.append(doc_qu_cnt)
#     return np.array(qc)
