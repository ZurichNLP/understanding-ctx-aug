#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import List, Dict
import random
import evaluate # https://huggingface.co/docs/evaluate/a_quick_tour

try:
    from .tokenization import tokenize_texts
except ImportError:
    from tokenization import tokenize_texts

# set seed for reproducibility
seed = 42
random.seed(seed)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(os.path.basename(__file__))

bleu = evaluate.load('sacrebleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
exact_match_metric = evaluate.load("exact_match")
bertscore = evaluate.load('bertscore')
chrf = evaluate.load('chrf')

def compute_rouge(predictions: List[str], references: List[List[str]], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/rouge

    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    """
    if verbose:
        # logger.info(f'Computing ROUGE with {len(predictions)} predictions and {len(references)} references...')
        print(f'Computing ROUGE with {len(predictions)} predictions and {len(references)} references...')
    
    if not is_tokenized:
        try:
            predictions = tokenize_texts(predictions)
            references = tokenize_texts(references)
        except AssertionError:
            logging.warning(f'Failed to tokenize texts. Computing ROUGE without tokenization.')

    return rouge.compute(predictions=predictions, references=references)

def compute_bleu(predictions: List[str], references: List[List[str]], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/sacrebleu
    
    predictions = ["hello there", "general kenobi"]
    references = [
        ["hello there general kenobi", "hello there !"],
        ["foo bar foobar"]
    ]   
    """ 
    if verbose:
        # logger.info(f'Computing BLEU with {len(predictions)} predictions and {len(references)} references...')
        print(f'Computing BLEU with {len(predictions)} predictions and {len(references)} references...')
    
    # sacrebleu applies tokenization to the predictions and references separately, 
    # but can override this behavior by passing setting force=True
    return bleu.compute(predictions=predictions, references=references, tokenize='13a', force=is_tokenized)

def compute_meteor(predictions: List[str], references: List[List[str]], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/meteor

    predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    references = [['It is a guide to action that ensures that the military will forever heed Party commands', 'It is the guiding principle which guarantees the military forces always being under the command of the Party', 'It is the practical guide for the army always to heed the directions of the party']]
    """
    if verbose:
        # logger.info(f'Computing METEOR with {len(predictions)} predictions and {len(references)} references...')
        print(f'Computing METEOR with {len(predictions)} predictions and {len(references)} references...')
    
    return meteor.compute(predictions=predictions, references=references)

def compute_exact_match(
    predictions: List[str], 
    references: List[str], 
    is_tokenized: bool = False, 
    regexes_to_ignore: List[str] = None, # passed directly to exact_match_metric.compute()
    ignore_case: bool = False, # passed directly to exact_match_metric.compute()
    ignore_punctuation: bool = False, # passed directly to exact_match_metric.compute()
    ignore_numbers: bool = False, # passed directly to exact_match_metric.compute()
    verbose: bool = False) -> Dict:
    """
    Compute exact match between predictions and references.
    """
    if verbose:
        # logger.info(f'Computing exact match with {len(predictions)} predictions and {len(references)} references...')
        print(f'Computing exact match with {len(predictions)} predictions and {len(references)} references...')

    return exact_match_metric.compute(
        predictions=predictions, 
        references=references, 
        regexes_to_ignore=regexes_to_ignore,
        ignore_case=ignore_case,
        ignore_punctuation=ignore_punctuation,
        ignore_numbers=ignore_numbers,
        )

def compute_self_bleu(predictions: List[str], use_subset: int = 200, is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    measures the similarity between each model prediction and all other predictions. 

    :use_subset: if set to a positive integer, only a subset of the predictions will be used as 'references' to compute the self-BLEU score.
    """
    references = []
    for i in range(len(predictions)):
        cur_refs = predictions[:i] + predictions[i+1:]
        assert len(cur_refs) == len(predictions) - 1
        cur_refs = random.sample(cur_refs, min(use_subset, len(cur_refs)))
        references.append(cur_refs)
    return compute_bleu(predictions, references, is_tokenized=is_tokenized, verbose=verbose)

def compute_bertscore(predictions: List[str], references: List[str], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/bertscore

    predictions = ["hello there", "general kenobi"]
    references = [
        ["hello there general kenobi", "hello there !"],
        ["foo bar foobar"]
    ]   
    """ 
    if verbose:
        # logger.info(f'Computing BERTScore with {len(predictions)} predictions and {len(references)} references...')
        print(f'Computing BERTScore with {len(predictions)} predictions and {len(references)} references...')
    return bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")

def compute_chrf(predictions: List[str], references: List[str], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/chrf

    predictions = ["hello there", "general kenobi"]
    references = [
        "hello there general kenobi",
        "foo bar foobar"
    ]   
    """ 
    if verbose:
        # logger.info(f'Computing chrF with {len(predictions)} predictions and {len(references)} references...')
        print(f'Computing chrF with {len(predictions)} predictions and {len(references)} references...')
    return chrf.compute(predictions=predictions, references=references)