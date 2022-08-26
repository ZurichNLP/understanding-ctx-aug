#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python test_run.py /scratch/tkew/ctrl_tokens/resources/models/muss_en_mined_hf/ [attn bias for prefix] [attn bias for control code]

python test_run.py /scratch/tkew/ctrl_tokens/resources/models/muss_en_mined_hf 5 5
This is very hard to understand. Is it like you are?
Dinosaurs used to travel the earth looking for food.

python test_run.py /scratch/tkew/ctrl_tokens/resources/models/muss_en_mined_hf 5 10
This is very difficult to understand. Are you a fan of?
Dinosaurs used to roam the earth looking for food. Are you going to be a dinosaur?

"""

import sys
from typing import List, Tuple, Dict, Optional
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForConditionalGeneration
import torch

def generate(inputs, model, tokenizer, beam_size=5, verbose=False):
    '''run generation with a hugging face model'''
    if verbose:
        print(f'generation inputs: {inputs}')
    # inputs = tokenizer(sentences, padding=True, return_tensors='pt').to(model.device)
    # check_uniqueness(input_ids['input_ids'])
    
    kwargs = {
        'cross_attention_bias': inputs["cross_attention_bias"].to(model.device) if "cross_attention_bias" in inputs else None,
        'context_code': inputs["context_code"].to(model.device) if "context_code" in inputs else None,
    }

    outputs_ids = model.generate(
        inputs["input_ids"].to(model.device), 
        attention_mask=inputs["attention_mask"].to(model.device) if "attention_mask" in inputs else None,
        num_beams=beam_size,
        num_return_sequences=1, 
        max_length=128,
        decoder_kwargs=kwargs,
        )
    outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs

def sanity_check_attn_bias(inputs, model, tokenizer):
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
        attn_weight = 1

    if len(sys.argv) > 3:
        context_attn_weight = int(sys.argv[3])
    else:
        context_attn_weight = 1

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForConditionalGeneration.from_pretrained(model_dir)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    prefix = '<DEPENDENCYTREEDEPTHRATIO_0.8> <WORDRANKRATIO_0.75> <REPLACEONLYLEVENSHTEIN_0.65> <LENGTHRATIO_0.75>'
    sentence1 = 'This is extremely complicated to comprehend.'
    sentence2 = 'Many moons ago, dinosaurs roamed the earth searching for their prey.'

    prefix_len = len(tokenizer.tokenize(prefix))+1 # add 1 to account for '<s>'
    inputs = tokenizer([prefix + sentence1, prefix + sentence2], padding=True, return_tensors='pt')

    ###################
    # attention biasing
    ###################
    # sanity_check_attn_bias(inputs, model, tokenizer)
    # breakpoint()
    
    # cross_attention_bias = inputs['attention_mask'].clone()
    # cross_attention_bias[:, 1:prefix_len] = attn_weight
    # outputs = generate({"input_ids": inputs["input_ids"], "cross_attention_bias": cross_attention_bias}, model, tokenizer, beam_size=5, verbose=True)
    # for output in outputs:
    #     print(output)

    ######################
    # context augmentation
    ######################
    context_examples = [
        'Am I a teacher?',
        'Are you from France?',
        'Is she tall?',
        'Do you like pizza?',
        'Does he have a brother?',
        'Did you eat breakfast this morning?',
        'Is it going to rain tonight?',
        'Were you on holidays last week?',
        'Will you attend university next year?',
        'Was he nice?',
    ]

    context_inputs = tokenizer(context_examples, padding=True, return_tensors='pt')    
    # get context code
    context_code = model.get_encoder()(context_inputs['input_ids'].to(model.device), return_dict=True)['last_hidden_state'].mean(dim=0).unsqueeze(dim=0)
    
    # copy context code for all items in batch (TODO: allow for different context code for each item)
    context_code = context_code.repeat(inputs['attention_mask'].size()[0], 1, 1)

    context_code_attention_mask = torch.ones([1, context_code.size()[1]], dtype=int) * context_attn_weight
    context_code_attention_mask = context_code_attention_mask.repeat(inputs['attention_mask'].size()[0], 1)

    outputs = generate({"input_ids": inputs["input_ids"], "context_code": context_code}, model, tokenizer, beam_size=5, verbose=False)
    for output in outputs:
        print(output)

    cross_attention_bias = inputs['attention_mask'].clone()
    cross_attention_bias[:, :prefix_len] = attn_weight
    cross_attention_bias = torch.cat([context_code_attention_mask, cross_attention_bias], dim=-1)

    outputs = generate({"input_ids": inputs["input_ids"], "cross_attention_bias": cross_attention_bias, "context_code": context_code}, model, tokenizer, beam_size=5, verbose=False)
    for output in outputs:
        print(output)
