#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

from evaluation.perplexity import score_ppl

test_set = "resources/data/Topical-Chat/KGD/test_freq.json"

def load_json_dataset(dataset: Path):   
    split = dataset.stem
    extension = dataset.suffix.strip('.')
    print(split, extension, str(dataset))
    dataset_dict = load_dataset(extension, data_files={'test': str(dataset)})
    return dataset_dict # concatenate_datasets(loaded_datasets)

def remove_speaker_ids(text):
    text = re.sub(r'<speaker[0-9]>\s+', '', text)
    return text


d = load_json_dataset(Path(test_set))

texts = d['test']['target']
texts = [remove_speaker_ids(text) for text in texts]

print(f'Original texts: {len(texts)}')
print(texts[:5])


# score texts in d['test']['target'] for ppl
ppl = score_ppl(texts)
print(f'PPL: {ppl}')