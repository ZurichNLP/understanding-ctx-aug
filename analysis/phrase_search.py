#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given a file with a list of phrases, get the frequency occurence of each phrase in the corpus.

Usage:
    python phrase_search.py <corpus_file> <phrase_file>
"""

import sys
from pprint import pprint
from collections import Counter

infile = sys.argv[1]
phrases_file = sys.argv[2]

def iter_lines(file):    
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            yield line.strip()

phrase_counter = Counter()
for line in iter_lines(phrases_file):    
    phrase_counter[line.lower()] = 0

for line in iter_lines(infile):
    for phrase in phrase_counter:
        if phrase in line.lower():
            phrase_counter[phrase] += 1

pprint(phrase_counter)
print(sum(phrase_counter.values()))

