#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python test_run.py /scratch/tkew/ctrl_tokens/resources/models/muss_en_mined_hf/
"""

import sys
from typing import List, Tuple, Dict, Optional
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def generate(inputs, model, tokenizer, beam_size=5, verbose=False):
    '''run generation with a hugging face model'''
    if verbose:
        print(f'generation inputs: {inputs}')
    # inputs = tokenizer(sentences, padding=True, return_tensors='pt').to(model.device)
    # check_uniqueness(input_ids['input_ids'])
    # breakpoint()
    outputs_ids = model.generate(
        inputs["input_ids"].to(model.device), 
        attention_mask=inputs["attention_mask"].to(model.device) if "attention_mask" in inputs else None,
        num_beams=beam_size,
        num_return_sequences=1, 
        max_length=128,
        decoder_kwargs={'cross_attention_bias': inputs["cross_attention_bias"].to(model.device)} if "cross_attention_bias" in inputs else None,
        # encoder_kwargs={'cross_attention_bias': inputs["cross_attention_bias"].to(model.device)} if "cross_attention_bias" in inputs else None,
        )
    outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs

def sanity_check(inputs, model, tokenizer):
    '''
    using an attention mask to mask out the prefix should yield 
    the same output as setting cross attnetion bias to 0 for the prefix
    '''
    input_tensor = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # regular inference with model
    outputs = generate({"input_ids": input_tensor}, model, tokenizer, beam_size=5, verbose=True)
    for output in outputs:
        print(output)

    # inference with attention mask on prefix (i.e. ignore the prefix)
    cust_attention_mask = inputs['attention_mask'].clone()
    cust_attention_mask[:, 1:prefix_len] = 0
    outputs = generate({"input_ids": input_tensor, "attention_mask": cust_attention_mask}, model, tokenizer, beam_size=5, verbose=True)
    for output in outputs:
        print(output)

    cross_attention_bias = inputs['attention_mask'].clone()
    cross_attention_bias[:, 1:prefix_len] = 0    
    outputs = generate({"input_ids": input_tensor, "cross_attention_bias": cross_attention_bias}, model, tokenizer, beam_size=5, verbose=True)
    for output in outputs:
        print(output)


if __name__ == '__main__':
    model_dir = sys.argv[1]

    if len(sys.argv) > 2:
        attn_weight = int(sys.argv[2])
    else:
        attn_weight = 0

    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    prefix = '<DEPENDENCYTREEDEPTHRATIO_0.8> <WORDRANKRATIO_0.75> <REPLACEONLYLEVENSHTEIN_0.65> <LENGTHRATIO_0.75>'
    sentence = 'This is extremely complicated to comprehend.'

    prefix_len = len(tokenizer.tokenize(prefix))+1 # add 1 to account for '<s>'
    inputs = tokenizer([prefix + sentence], padding=True, return_tensors='pt')

    sanity_check(inputs, model, tokenizer)

    cross_attention_bias = inputs['attention_mask'].clone()
    cross_attention_bias[:, 1:prefix_len] = attn_weight
    outputs = generate({"input_ids": inputs["input_ids"], "cross_attention_bias": cross_attention_bias}, model, tokenizer, beam_size=5, verbose=True)
    for output in outputs:
        print(output)

    