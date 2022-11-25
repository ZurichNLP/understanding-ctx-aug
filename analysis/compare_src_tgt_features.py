#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

from sim_metrics import compute_sim

df = pd.read_csv(sys.argv[1], sep=',') # resources/sentiment_features.csv
feature_type = sys.argv[2] # 'sentiment'

print(f'Correlation between src and tgt texts given the existance of a specific feature: {feature_type}')
print(df.columns)

# df = df.sample(frac=0.25).reset_index(drop=True) # shuffle
# print(df.shape)

for col in df.columns:
    if col.startswith('src') and col.endswith(feature_type):
        print(f'{col} vs. TGT')
        compute_sim(df[col], df[f'tgt_{feature_type}'])

# speaker consistency
print('\nSpeaker consistency')
spkr_preds_summed = df[[f'src_2_{feature_type}', f'src_4_{feature_type}']].to_numpy().sum(axis=1)
for i in range(1, max(spkr_preds_summed) + 1):
    print(f'SRC (>={i}) vs. TGT')
    src_ind = [1 if x >= i else 0 for x in spkr_preds_summed]
    compute_sim(src_ind, df[f'tgt_{feature_type}'])

# mirroring
print('\nMirroring')
opp_spkr_preds_summed = df[[f'src_1_{feature_type}', f'src_3_{feature_type}',f'src_5_{feature_type}']].to_numpy().sum(axis=1)
for i in range(1, max(opp_spkr_preds_summed) + 1):
    print(f'SRC (>={i}) vs. TGT')
    src_ind = [1 if x >= i else 0 for x in opp_spkr_preds_summed]
    compute_sim(src_ind, df[f'tgt_{feature_type}'])

# strength of src sentiment features
print(f'\nStrength of src {feature_type} features')
src_preds_summed = df[[f'src_0_{feature_type}', f'src_1_{feature_type}', f'src_2_{feature_type}', f'src_3_{feature_type}', f'src_4_{feature_type}', f'src_5_{feature_type}']].to_numpy().sum(axis=1)
for i in range(1, max(src_preds_summed) + 1):
    src_ind = [1 if x >= i else 0 for x in src_preds_summed]
    print(f'SRC (>={i}) vs. TGT')
    compute_sim(src_ind, df[f'tgt_{feature_type}'])

# for i in range(1, max(src_preds_summed) + 1):
print(f'Majority')
# breakpoint()
src_ind = [1 if x >= 3 else 0 for x in src_preds_summed]
compute_sim(src_ind, df[f'tgt_{feature_type}'])

