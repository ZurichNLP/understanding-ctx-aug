#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# these are the hedging expressions from the LREC 2020 paper
# https://github.com/hedging-lrec/resources/blob/master/discourse_markers.txt
# the detection method is based on simple string matching and is not perfect!

import sys

hedging_contrast = [
    "however",
    "on the one hand",
    "on the other hand",
    "rather",
    "in contrast",
    "nonetheless",
    "though",
    "all the same",
    "anyway",
    "as a matter of fact",
    "at the same time",
    "perhaps",
]

hedging_management = [
    "actually",
    "anyway",
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
]

def detect_hedging(text, hedging_list):
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

def count_hedges(texts, hedging_list):
    """Counts the number of hedging expressions in a list of texts.

    Args:
        texts (list): List of texts to be analyzed.
        hedging_list (list): List of hedging expressions.

    Returns:
        int: Number of hedging expressions in the texts.
    """
    
    hedge_count = sum([detect_hedging(text, hedging_list) for text in texts])
    
    return hedge_count


if __name__ == "__main__":
    
    
    texts = [
        "however, it seems rough!",
        "I think that's not really easy to say.",
        "I do not want to answer this question.",
        "taht's bad",
    ] 
    
    print(detect_hedging(texts[0], hedging_contrast))
    print(detect_hedging(texts[1], hedging_management))
    print(detect_hedging(texts[2], hedging_evasion))
    
    
    infile = sys.argv[1]

    with open(infile, "r", encoding="utf8") as f:
        texts = f.readlines()
    print(type(texts))
    print(len(texts))
    print(texts[0])
    print(count_hedges(texts, hedging_contrast) / len(texts))
    print(count_hedges(texts, hedging_management) / len(texts))
    print(count_hedges(texts, hedging_evasion) / len(texts))
    
