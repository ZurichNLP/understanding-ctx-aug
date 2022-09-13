#!/usr/bin/env bash
# -*- coding: utf-8 -*-

checkpoint_dir=$1 # resources/models/seed_42/pt/fairseq/roberta_small-MLM/
data_dir=$2 # resources/data/books1/bin
tokenizer_dir=$3 # data_dir/../tok/tokenizer/
output_dir=$4 # resources/models/seed_42/pt/hf_conv/roberta_small-MLM

{ [ -z "$checkpoint_dir" ] || [ -z "$data_dir" ] || [ -z "$tokenizer_dir" ] || [ -z "$output_dir" ]; } && echo "Usage: convert_roberta.sh checkpoint_dir data_dir tokenizer_dir output_dir" && exit 1

# rename checkpoint for conversion script (expects `model.pt`)
mv "$checkpoint_dir/checkpoint_last.pt" "$checkpoint_dir/model.pt"

# conversion script require fairseq's dict.txt to be in the checkpoint dir
cp "$data_dir/dict.txt" "$checkpoint_dir"

python pretraining/convert_fairseq_roberta_model_to_transformers.py \
    --roberta_checkpoint_path "$checkpoint_dir" \
    --pytorch_dump_folder_path "$output_dir" \

# copy over original tokenizer as well
cp "$tokenizer_dir"/* "$output_dir"