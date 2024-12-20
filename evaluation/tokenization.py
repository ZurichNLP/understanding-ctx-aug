#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Optional, Union
from tqdm import tqdm
from mosestokenizer import MosesSentenceSplitter, MosesTokenizer

def sentencize_texts(texts: List[str], lang: str = 'en', verbose=True) -> List[List[str]]:
    """
    Split 'texts' at sentence boundaries.
    """
    tok_texts = []
    # breakpoint()
    with MosesSentenceSplitter(lang) as splitsents:
        if verbose:
            for text in tqdm(texts, total=len(texts), desc="Splitting Sentences"):
                if text:
                    sentences = splitsents([text]) # note, MosesSentenceSplitter takes a list of lines (strings) and returns a list of sentences (strings)
                    tok_texts.append(sentences)
                else:
                    tok_texts.append([''])
        else:
            for text in texts:
                if text:
                    sentences = splitsents([text])
                    tok_texts.append(sentences)
                else:
                    tok_texts.append([''])
    return tok_texts

def tokenize_ref_texts(texts: List[str], lang: str = 'en'):
    """
    Tokenize references for metrics in Hugging Face evaluation package.

    :texts: are expected to be a list of lists of strings, where each list of strings is a reference.
    """
    tok_texts = []
    with MosesTokenizer(lang) as tokenize:
        for text in texts:
            tok_text = []
            for text_i in text:
                tok_text.append(' '.join(tokenize(text_i)))
            tok_texts.append(tok_text)
    return tok_texts

def tokenize_texts(texts: Union[List[str], List[List[str]]], lang: str = 'en'):
    """
    Tokenize texts for metrics in Hugging Face evaluation package.

    :texts: are expected to be either a list of lists of strings (where each list of strings is a reference)
    or a list of strings (where each string is a sys output).
    """
    if isinstance(texts[0], list):
        return tokenize_ref_texts(texts, lang) 
    else:
        with MosesTokenizer(lang) as tokenize:
            return [' '.join(tokenize(text)) for text in texts]
