#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re

infile = 'resources/data/Topical-Chat/KGD/train.json'

for max_qus in range(0, 5):
    outfile = f'resources/data/Topical-Chat/KGD/train_max{max_qus}qus.json'
    c = 0

    with open(infile, 'r', encoding='utf8') as inf:
        with open(outfile, 'w', encoding='utf8') as outf:
            for line in inf:
                line_question_count = len(re.findall(r'\?', line))
                # breakpoint()
                if line_question_count > max_qus:
                    pass
                else:
                    outf.write(line)
                    c += 1

    print(f'Wrote {c} lines to {outfile}')
