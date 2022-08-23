#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import List, Optional, Dict
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", required=False, nargs='*', help="Files to train tokenizer on")
    ap.add_argument("--path_to_tokenizer", required=False, type=str, help="Path to tokenizer to learn/apply")
    ap.add_argument("--vocab_size", required=False, type=int, default=2048, help="Vocab size")
    ap.add_argument("--min_frequency", required=False, type=int, default=2, help="Minumin frequency")
    ap.add_argument("--from_existing", required=False, default="facebook/bart-base", type=str, help="Initialise from an existing tokenizer, e.g. BART")
    ap.add_argument("--overwrite", required=False, action='store_true', help="Overwrite an existing ouput tokenizer if one already exists")
    ap.add_argument("--batch_size", required=False, type=int, default=1000, help="Batch size for tokenizer training iterator")
    return ap.parse_args()

def load_text_dataset(train_file: str):
    """load dataset from text file"""
    dataset = load_dataset("text", data_files={"train": [train_file]})['train']
    return dataset

def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]

def batch_iterator(dataset, column, batch_size: int = 1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size][column]

def save_tokenizer(tokenizer, out_path: str, overwrite: bool = False):
    
    if Path(out_path).exists() and not overwrite:
        raise RunTimeError(f"[!] Tokenizer already exists at {out_path}. Overwrite existing tokenizer with --overwrite")

    Path(out_path).mkdir(parents=True, exist_ok=True)

    if isinstance(tokenizer, ByteLevelBPETokenizer):    
        # <class 'tokenizers.implementations.byte_level_bpe.ByteLevelBPETokenizer'> do not have a save_pretrained method
        tokenizer.save(str(Path(out_path) / "tokenizer.json"))
    else:
        tokenizer.save_pretrained(str(Path(out_path)))
    
    print('Tokenizer:')
    print(tokenizer)
    print(f"Saved to {out_path}")

    return

def train_tokenizer(
    dataset,
    vocab_size: int = 2048,
    min_frequency: int = 2,
    batch_size: int = 1000,
    from_existing: Optional[str] = None,
    ):

    special_tokens = {
        'bos_token': '<s>', 
        'pad_token': '<pad>', 
        'eos_token': '</s>', 
        'unk_token': '<unk>', 
        # 'sep_token': '</s>', 
        # 'cls_token': '<s>', 
        # 'mask_token': '<mask>'
    }

    if not from_existing:
        # Instantiate tokenizer
        tokenizer = ByteLevelBPETokenizer()

        # Customized training
        tokenizer.train_from_iterator(
            batch_iterator(dataset, "text", batch_size), 
            vocab_size=vocab_size, 
            min_frequency=min_frequency, 
            special_tokens=list(special_tokens.values())
            )

    else:

        # Load existing tokenizer
        tokenizer = AutoTokenizer.from_pretrained(from_existing)

        # Retrain existing tokenizer
        tokenizer = tokenizer.train_new_from_iterator(
            batch_iterator(dataset, "text", batch_size), 
            vocab_size=vocab_size, 
            special_tokens_map=special_tokens, # ensure that BART's special tokens are in the vocabulary, e.g. this is not the case if initializing from e.g. gpt2
            )

    return tokenizer

# # https://github.com/huggingface/tokenizers/issues/640#issuecomment-792305076
# def bpe_tokenizer_trainer(text, vocab_size, min_frequency=0, add_prefix_space=True, batch_size=50):
#     # Supply either path to txt file or list of strings as text arg

#     tokenizer = Tokenizer(models.BPE())

#     tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
#         [
#             pre_tokenizers.Whitespace(),
#             pre_tokenizers.Punctuation(),
#             pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space),
#         ]
#     )
#     tokenizer.normalizer = normalizers.Sequence(
#         [normalizers.Nmt(), normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " "),]
#     )

#     tokenizer.decoder = decoders.ByteLevel()

#     trainer = trainers.BpeTrainer(
#         vocab_size=vocab_size,
#         special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
#         min_frequency=min_frequency,
#         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
#     )

#     if isinstance(text, str):
#         # if user specified path to txt file as string
#         tokenizer.train(text, trainer=trainer)
#     else:
#         # text is a datasets Dataset
#         tokenizer.train_from_iterator(batch_iterator(text, len(text), batch_size), trainer=trainer)

#     tokenizer.post_processor = processors.RobertaProcessing(
#         sep=("</s>", tokenizer.token_to_id("</s>")), cls=("<s>", tokenizer.token_to_id("<s>"))
#     )

#     tokenizer.save("tokenizer.json", pretty=True)
#     # tokenizer.model.save("output_dir")

def main(args):

    dataset = concatenate_datasets([load_text_dataset(f) for f in args.train_data])
    tokenizer = train_tokenizer(
        dataset=dataset,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        batch_size=args.batch_size,
        from_existing=args.from_existing,
        )
    
    save_tokenizer(tokenizer, args.path_to_tokenizer, args.overwrite)


if __name__ == "__main__":
    args = set_args()
    main(args)