#!/usr/bin/env bash
# -*- coding: utf-8 -*-

checkpoint_dir=$1
tokenizer=$2
output_dir=$3

{ [ -z "$checkpoint_dir" ] || [ -z "$tokenizer" ] || [ -z "$output_dir" ]; } && echo "Usage: convert_model.sh checkpoint_dir tokenizer output_dir" && exit 1

python pretraining/convert_fairseq_bart_model_to_transformers.py \
    --checkpoint "$checkpoint_dir/checkpoint_best.pt" \
    --tokenizer "$tokenizer" \
    --output_dir "$output_dir"