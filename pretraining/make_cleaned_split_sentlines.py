#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted from https://github.com/soskek/bookcorpus/blob/master/make_sentlines.py
"""

import os
import sys
import random
from glob import glob
import string
from pathlib import Path
import logging
from blingfire import text_to_sentences

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
sent_final_punctuation = [".", "!", "?"]

def ivalid_char_ratio(text):
    char_count = sum(1 if char in string.punctuation else 0 for char in text)
    num_count = sum(1 if char.isdigit() else 0 for char in text)
    count = char_count + num_count
    spaces = text.count(" ")
    total_chars = len(text) - spaces
    return count / total_chars

def validate_sent(text):
    text = text.strip()
    if not text:
        return None
    elif text[-1] not in sent_final_punctuation:
        return None
    elif text.startswith('#'):
        return None
    elif ivalid_char_ratio(text) > 0.7:
        return None
    else:
        return text


def convert_into_sentences(lines):
    stack = []
    sent_L = []
    n_sent = 0
    for chunk in lines:
        if not chunk.strip():
            if stack:
                sents = text_to_sentences(
                    " ".join(stack).strip().replace('\n', ' ')).split('\n')
                sent_L.extend(sents)
                n_sent += len(sents)
                sent_L.append('\n')
                stack = []
            continue
        stack.append(chunk.strip())

    if stack:
        sents = text_to_sentences(
            " ".join(stack).strip().replace('\n', ' ')).split('\n')
        sent_L.extend(sents)
        n_sent += len(sents)
    return sent_L, n_sent

def write_sentences_to_file(file_list, out_file):
    with open(out_file, 'w', encoding='utf8') as outf:
        total_n_sent = 0
        for i, file_path in enumerate(file_list):
            sents, _ = convert_into_sentences(open(file_path).readlines())
            # filter for 'decent' looking text
            sents = list(filter(validate_sent, sents))
            n_sent = len(sents)
            total_n_sent += n_sent
            outf.write('\n'.join(sents) + '\n')
            logger.info(f'{i}/{len(file_list)}\t{n_sent}\t{out_file}')
    
    logger.info(f'*** wrote {total_n_sent} sentences from {len(file_list)} files ***')
    return


if __name__ == '__main__':
    
    file_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    test_size = int(sys.argv[3])

    out_dir.mkdir(exist_ok=True, parents=True)
    test_file = out_dir / 'valid.txt'
    train_file = out_dir / 'train.txt'
    
    file_list = list(sorted(file_dir.glob('*.epub.txt')))
    random.seed(42)
    random.shuffle(file_list)
    
    test_file_list = file_list[:test_size]
    train_file_list = file_list[test_size:]
    assert len(set(test_file_list).intersection(set(train_file_list))) == 0

    write_sentences_to_file(test_file_list, test_file)
    write_sentences_to_file(train_file_list, train_file)
