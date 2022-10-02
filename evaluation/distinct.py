#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
intra_dist = avg. sequence level distinctiveness (within a sequence)
inter_dist = corpus level ngram distinctiveness (between sequences) (i.e. ngram variation in model outputs)
"""

from typing import List
from collections import Counter
import numpy as np
from tqdm import tqdm

def distinct_n(seqs: List[str]):
    """ Calculate intra/inter distinct 1/2. """
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in tqdm(seqs, total=len(seqs), desc="Computing Distinct-N"):
        if isinstance(seq, str):
            seq = seq.split()
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return {
        'intra_dist1': intra_dist1, 
        'intra_dist2': intra_dist2, 
        'inter_dist1': inter_dist1,
        'inter_dist2': inter_dist2,
    }
