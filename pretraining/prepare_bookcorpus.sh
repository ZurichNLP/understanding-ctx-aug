#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md

# exit on error
set -e

############
# PARSE ARGS
############

while getopts "h?fd:" opt; do
  case "$opt" in
    h|\?)
      echo "Usage: bash $0 -d data_dir [-f]" && exit 0
        ;;
    f) FORCE_OVERWRITE="1"
        ;;
    d) DATA_DIR=$OPTARG
        ;;
  esac
done

[[ -z "$DATA_DIR" ]] && echo "data not provided, exiting" && exit 1

RAW="$DATA_DIR/raw"
BIN="$DATA_DIR/bin"
TOK="$DATA_DIR/tok"

# Run these checks first to make sure we're not unintentionally overwriting anything
{ [ -d "$RAW" ] && [ -z "$FORCE_OVERWRITE" ]; } && echo "tmp data exists, exiting" && exit 1
{ [ -d "$TOK" ] && [ -z "$FORCE_OVERWRITE" ]; } && echo "tok data exists, exiting" && exit 1
{ [ -d "$BIN" ] && [ -z "$FORCE_OVERWRITE" ]; } && echo "bin data exists, exiting" && exit 1

rm -rf "$RAW" "$TOK" "$BIN" && mkdir -p "$RAW" "$TOK" "$BIN"

# extract sentences from books 
# script outputs train.txt and valid.txt files
echo ""
echo "Extracting sentences from books..."
echo ""
python make_cleaned_split_sentlines.py \
    "$DATA_DIR/epubtxt/" \
    "$RAW" \
    100

# train tokenizer
echo ""
echo "Training tokenizer..."
echo ""
python tokenizers_trainer.py \
    --train_data "$RAW/train.txt" \
    --path_to_tokenizer "$TOK/tokenizer" \
    --from_existing "facebook/bart-base" \
    --vocab_size 4096 --min_frequency 5 \
    --overwrite

echo ""
echo "Applying tokenizer..."
echo ""
for split in train valid; do
    python tokenizers_encode.py \
        -i "$RAW" \
        -o "$TOK" \
        -f "${split}.txt" \
        --path_to_tokenizer "$TOK/tokenizer"
done

echo ""
echo "Converting tokenizer for Fairseq..."
echo ""
python convert_tokenizer_to_fairseq_dict.py -v "$TOK/tokenizer/vocab.json" -o "$TOK"

# --only-source ensures that preprocessing is only done once and that 
# the provided dict.txt is used for preprocessing. Without this argument
# dict.txt is overwritten and files are preprocessed twice.
echo ""
echo "Binarizing tokenized data for Fairseq..."
echo ""
fairseq-preprocess \
    --trainpref "$TOK/train.tok" \
    --validpref "$TOK/valid.tok" \
    --only-source --srcdict "$TOK/dict.txt" \
    --task "denoising" \
    --dataset-impl "mmap" \
    --destdir "$BIN" \
    --workers 8

cp "$TOK/dict.txt" "$BIN"

echo ""
echo "Done!"
echo ""
