#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import List, Dict

import evaluate # https://huggingface.co/docs/evaluate/a_quick_tour

try:
    from .tokenization import tokenize_texts
except ImportError:
    from tokenization import tokenize_texts

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(os.path.basename(__file__))

bleu = evaluate.load('sacrebleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

def compute_rouge(predictions: List[str], references: List[str], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/rouge

    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    """
    if verbose:
        logger.info(f'Computing ROUGE with {len(predictions)} predictions and {len(references)} references.')
        
    if not is_tokenized:
        try:
            predictions = tokenize_texts(predictions)
            references = tokenize_texts(references)
        except AssertionError:
            logging.warning(f'Failed to tokenize texts. Computing ROUGE without tokenization.')

    return rouge.compute(predictions=predictions, references=references)

def compute_bleu(predictions: List[str], references: List[str], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/sacrebleu
    
    predictions = ["hello there", "general kenobi"]
    references = [
        ["hello there general kenobi", "hello there !"],
        ["foo bar foobar"]
    ]   
    """ 
    if verbose:
        logger.info(f'Computing BLEU with {len(predictions)} predictions and {len(references)} references.')
    # sacrebleu applies tokenization to the predictions and references separately, 
    # but can override this behavior by passing setting force=True
    return bleu.compute(predictions=predictions, references=references, force=is_tokenized)

def compute_meteor(predictions: List[str], references: List[str], is_tokenized: bool = False, verbose: bool = False) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/meteor

    predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    references = [['It is a guide to action that ensures that the military will forever heed Party commands', 'It is the guiding principle which guarantees the military forces always being under the command of the Party', 'It is the practical guide for the army always to heed the directions of the party']]
    """
    if verbose:
        logger.info(f'Computing METEOR with {len(predictions)} predictions and {len(references)} references.')
    return meteor.compute(predictions=predictions, references=references)
