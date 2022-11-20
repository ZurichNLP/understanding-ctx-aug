#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# these are the hedging expressions from the LREC 2020 paper
# https://github.com/hedging-lrec/resources/blob/master/discourse_markers.txt
# the detection method is based on simple string matching and is not perfect!

"""

This script implements the hedging detection method from the LREC 2020 paper and the HedgeHog Model from

Example Usage:

python analysis/hedge.py resources/data/Topical-Chat/KGD/test_freq.tgt
Counter({'C': 180341, 'E': 1548, 'D': 1370, 'N': 984, 'I': 324})

"""

import sys

import torch
from simpletransformers.ner import NERModel
from collections import Counter

hedging_contrast = [
    "however",
    "on the one hand",
    "on the other hand",
    # "rather",
    "in contrast",
    "nonetheless",
    "although",
    "even though",
    # "though",
    "whereas",
    "while",
    "on the contrary",
    "all the same",
    # "anyway",
    "as a matter of fact",
    "at the same time",
    "conversely",
    # "perhaps",
]

hedging_management = [
    # "actually",
    # "anyway",
    "i would say",
    "so well",
    "you know",
    "you see",
    "sort of",
    "kind of",
    "so to speak",
    "more or less",
    "not really",
    "no real instance",
    "difficult question",
    "difficult to answer",
    "hard to say",
    "hard to answer",
]

hedging_evasion = [
    "i do not want to",
    "i don't want to",
    "i am not going to",
    "i ain't going to",
    "i am not trying to",
    "i ain't trying to",
    "i will not say",
    "i won't say",
    "i will not mention",
    "i won't mention",
    "i do not know",
    "i don't know",
    "i really do not know",
    "i really don't know",
    "i do not really understand",
    "i don't really understand",
    "i can not find the word",
    "i can't find the word",
    "i can not think of",
    "i can't think of",
    "i can not remember",
    "i can't remember",
    "i can not recall",
    "i can't recall",
    "i can not say",
    "i can't say",
    "i'd rather not",
]

def simple_search_detect_hedging(text, hedging_list):
    """Detects hedging in a text via simple string matching.

    Args:
        text (str): Text to be analyzed.
        hedging_list (list): List of hedging expressions.

    Returns:
        bool: True if the text contains a hedging expression, False otherwise.
    """
    for hedge in hedging_list:
        if hedge in text.lower():
            return True
    return False

def count_hedge_phrases(texts, hedging_list):
    """Counts the number of hedging expressions in a list of texts.

    Args:
        texts (list): List of texts to be analyzed.
        hedging_list (list): List of hedging expressions.

    Returns:
        int: Number of hedging expressions in the texts.
    """
    
    hedge_count = sum([simple_search_detect_hedging(text, hedging_list) for text in texts])
    
    return hedge_count

def read_lines(infile):
    lines = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
    return lines

class HedgeHogModel():

    def __init__(self, use_cuda=True, batch_size=256, mp=True):

        self.labels = ["C", "D", "E", "I", "N"]
        self.model = NERModel(
            'bert',
            'jeniakim/hedgehog',
            use_cuda=use_cuda, #torch.cuda.is_available(),
            labels=self.labels,
            args={
                'eval_batch_size': batch_size,
                'use_multiprocessing': mp,
                },
        )

        self.examples = [
            "It could be that I'm not the best person to ask about this, but I think that you should definitely go for it.",
            "Perhaps he's not around.",
            "She might be at work.",
            "I believe that Daniel Radcliffe is a ghost.",
            "Apparently that's true!",
            "Only if it's not too much trouble, we can do that.",
            "It's believed that JKF jnr is not dead.",
        ]

    def annotate(self, texts):
        predictions, raw_outputs = self.model.predict(texts)
        return (predictions, raw_outputs)

    def unpack_predictions(self, predictions):
        for text in predictions:
            tokens, labels = [], []
            for annotated_token in text:
                token, label = annotated_token.popitem()
                tokens.append(token)
                labels.append(label)
            yield (tokens, labels)

    def count_hedges(self, predictions):
        total_labels = Counter()
        sentence_labels = []
        for _, labels in self.unpack_predictions(predictions):
            total_labels.update(labels)
            
            labels = list(filter(lambda x: x != 'C', labels))
            if len(labels) > 0:
                sentence_labels.append(labels[0]) # take the first label
            else:
                sentence_labels.append('C')
        sentence_labels = Counter(sentence_labels) # convert to Counter
        return total_labels, sentence_labels

if __name__ == "__main__":
    
    
    # texts = [
    #     "however, it seems rough!",
    #     "I think that's not really easy to say.",
    #     "I do not want to answer this question.",
    #     "taht's bad",
    # ] 
    
    # print(simple_search_detect_hedging(texts[0], hedging_contrast))
    # print(simple_search_detect_hedging(texts[1], hedging_management))
    # print(simple_search_detect_hedging(texts[2], hedging_evasion))
    
    
    infile = sys.argv[1]
    texts = read_lines(infile)

    print(type(texts))
    print(len(texts))
    print(texts[0])

    print(count_hedge_phrases(texts, hedging_contrast) / len(texts))
    print(count_hedge_phrases(texts, hedging_management) / len(texts))
    print(count_hedge_phrases(texts, hedging_evasion) / len(texts))
    
    hdm = HedgeHogModel()
    predictions, _ = hdm.annotate(texts)
    total = hdm.count_hedges(predictions)
    print(total)
