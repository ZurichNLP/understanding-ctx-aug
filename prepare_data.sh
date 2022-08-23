#!/usr/bin/env bash
# -*- coding: utf-8 -*-

for split in valid_rare valid_freq test_freq test_rare train; do
    python prepare_topical_chat_dataset.py \
        --data_dir resources/data/Topical-Chat \
        --split $split \
        --save_dir resources/data/Topical-Chat/KGD
done