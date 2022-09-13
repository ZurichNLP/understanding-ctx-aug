#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted from https://github.com/kb-labb/kb_bart/blob/main/save_to_huggingface.py

Example usage:
    python convert_fairseq_bart_model_to_transformers.py \
        --checkpoint resources/models/pt/fairseq/bart_small/checkpoint_best.pt \
        --tokenizer resources/data/books1/tok/tokenizer \
        --output_dir resources/models/pt/huggingface_conv/bart_small
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerFast, BartConfig, BartForConditionalGeneration

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, default=None)
    parser.add_argument("--tokenizer", type=str, required=True, default=None)
    parser.add_argument("--output_dir", type=str, required=True, default=None)
    parser.add_argument("--base_config", type=str, required=False, default="facebook/bart-base")
    return parser.parse_args()

def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

def get_model_layer_counts(modules, verbose=False):
    encoder_layers = set()
    decoder_layers = set()
    for m in modules:
        if verbose:
            print(f'{m}: has shape {modules[m].shape}')
        name_parts = m.split(".")
        if name_parts[0] == "encoder" and name_parts[1] == "layers":
            try:
                encoder_layers.add(int(name_parts[2]))
            except ValueError:
                pass
        elif name_parts[0] == "decoder" and name_parts[1] == "layers":
            try:
                decoder_layers.add(int(name_parts[2]))
            except ValueError:
                pass
    return len(encoder_layers), len(decoder_layers)

def main():
    args = get_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # tokenizer = PreTrainedTokenizerFast(
    #     tokenizer_file=args.tokenizer,
    #     # bos_token="<s>",
    #     # eos_token="</s>",
    #     # unk_token="<unk>",
    #     # pad_token="<pad>",
    #     # mask_token="<mask>",
    #     # cls_token="</s>",
    #     # sep_token="</s>",
    # )

    state_dict = torch.load(args.checkpoint, map_location="cpu")["model"]
        
    vocab_size = state_dict["encoder.embed_tokens.weight"].size()[0]
    max_position_embeddings = state_dict["encoder.embed_positions.weight"].size()[0]
    encoder_ffn_dim = state_dict["encoder.layers.0.fc1.bias"].size()[0]
    decoder_ffn_dim = state_dict["decoder.layers.0.fc1.bias"].size()[0]
    # number of heads are not saved explicitly, so we need to infer it from the size of the weights
    encoder_attention_heads = encoder_ffn_dim // (max_position_embeddings - 2) # -2 for <s> and </s>
    decoder_attention_heads = decoder_ffn_dim // (max_position_embeddings - 2) # -2 for <s> and </s>
    d_model = state_dict["decoder.layers.0.fc2.bias"].size()[0]
    encoder_layers, decoder_layers = get_model_layer_counts(state_dict)
    
    # config = AutoConfig.from_pretrained(args.config)
    config = BartConfig.from_pretrained(args.base_config)
    config.update({
        "vocab_size": vocab_size,
        "encoder_attention_heads": encoder_attention_heads,
        "decoder_attention_heads": decoder_attention_heads,
        "d_model": d_model,
        "encoder_ffn_dim": encoder_ffn_dim,
        "decoder_ffn_dim": decoder_ffn_dim,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "max_position_embeddings": max_position_embeddings - 2, # -2 for <s> and </s>
        })

    model = BartForConditionalGeneration(config)

    remove_ignore_keys_(state_dict)
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    model.model.load_state_dict(state_dict)

    model.lm_head = make_linear_from_emb(model.model.shared)

    # Save to Huggingface format
    config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    print(f"Saved converted model to {args.output_dir}")

if __name__ == "__main__":
    main()