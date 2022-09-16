#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Optional, Union
from tqdm import tqdm
from collections import Counter

try:
    from .tokenization import tokenize_texts
except ImportError:
    from tokenization import tokenize_texts

def compute_novelty_ngram_ratios(
    predictions: List[str], 
    references: Union[List[str], List[List[str]]], 
    is_tokenized: bool = False, 
    N: int = 4, 
    ignore_case: bool = False, 
    verbose: bool = False
    ) -> Dict:
    """
    Compute novel ngram ratios between predictions and references.
    """
    if verbose:
        print(f'Computing novelty with {len(predictions)} predictions and {len(references)} references...')
    
    if not is_tokenized:
        references = tokenize_texts(references)
        predictions = tokenize_texts(predictions)

    if ignore_case: # lowercase all strings
        if isinstance(references[0], list): # handle embedded lists
            references = [tok_text.lower().split() for reference in references for tok_text in reference]
        else:
            references = [tok_text.lower().split() for tok_text in references]
        predictions = [tok_text.lower().split() for tok_text in predictions]
    else:
        if isinstance(references[0], list): # handle embedded lists
            references = [tok_text.split() for reference in references for tok_text in reference]
        else:
            references = [tok_text.split() for tok_text in references]
        predictions = [tok_text.split() for tok_text in predictions]
        
    novelty_ratios = {f'{i}_gram': [] for i in range(1, N+1)}
    for prediction, reference in zip(predictions, references):
        for n in range(1, N+1):
            reference_ngrams = Counter([' '.join(reference[i:i+n]) for i in range(len(reference)-n+1)])
            prediction_ngrams = Counter([' '.join(prediction[i:i+n]) for i in range(len(prediction)-n+1)])
            
            if not len(prediction_ngrams):
                novelty_ratio = 0.0            
            else:
                novel_tokens = set(prediction_ngrams) - set(reference_ngrams)
                novelty_ratio = len(novel_tokens) / len(prediction_ngrams)
            
            novelty_ratios[f'{n}_gram'].append(novelty_ratio)

    return novelty_ratios

def compute_novelty(
    predictions: List[str], 
    references: Union[List[str], List[List[str]]], 
    is_tokenized: bool = False, 
    N: int = 4, 
    ignore_case: bool = False, 
    verbose: bool = False
    ) -> Dict:

    novelty_ratios = compute_novelty_ngram_ratios(predictions, references, is_tokenized, N, ignore_case, verbose)
    average_novelty_ratio = {f'{i}_gram': sum(novelty_ratios[f'{i}_gram'])/len(novelty_ratios[f'{i}_gram']) for i in range(1, N+1)}
    return average_novelty_ratio

if __name__ == "__main__":

    predictions = ['This is a new prediction', 'help', 'this is a somewhat new prediction!']
    references = [['Most of the references look like this'], ['Help!'], ['this is a prediction!']]

    print(compute_novelty(predictions, references, is_tokenized=False, ignore_case=True))
    print(compute_novelty(predictions, [' '.join(r) for r in references], is_tokenized=False, ignore_case=True)) # list of strings for references
    print(compute_novelty(predictions, references, is_tokenized=False, ignore_case=False)) # case sensitive

    