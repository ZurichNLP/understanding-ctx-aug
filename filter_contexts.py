#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example Usage:
    python filter_contexts.py resources/data/Topical-Chat/KGD/contexts/train_questions.txt > resources/data/Topical-Chat/KGD/contexts/train_ambig_questions.txt

    then run

    sed 's/?/!/g' resources/data/Topical-Chat/KGD/contexts/train_ambig_questions.txt > resources/data/Topical-Chat/KGD/contexts/train_amibig_exclamations.txt
"""

import sys
import re

infile = sys.argv[1]

wh_qus = re.compile(r"\b(what|which|when|where|who|whom|whose|why|how|whoever)\b", re.IGNORECASE)
interro_qus = re.compile(r"\b(isn?'?t?|aren?'?t?|wasn?'?t?|weren?'?t?|don?'?t?|doesn?'?t?|didn?'?t?|haven?'?t?|hasn?'?t?|hadn?'?t?)\b", re.IGNORECASE)
aux_qus = re.compile(r"\b(can'?t?|couldn?'?t?|may|might|mustn?'?t?|shall|shouldn?'?t?|will|wouldn?'?t?|won'?t)\b", re.IGNORECASE)
qu_tags = re.compile(r'\b(right|huh|, you|and you)\b', re.IGNORECASE)
pron_init = re.compile(r'^(you|he|she|they)\b', re.IGNORECASE)

with open(infile, 'r', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        # breakpoint()
        if wh_qus.search(line):
            pass
        elif interro_qus.search(line):
            pass
        elif aux_qus.search(line):
            pass
        elif qu_tags.search(line):
            pass
        elif pron_init.search(line):
            pass
        else:
            print(line)
            
            # line = re.sub(r'\s+', ' ', line)
            # with open(outfile, 'a', encoding='utf8') as f:
            #     f.write(line + '


