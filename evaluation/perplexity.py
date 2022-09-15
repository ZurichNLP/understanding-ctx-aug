# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

from typing import List
import os
import numpy as np
import evaluate # https://huggingface.co/docs/evaluate/a_quick_tour

os.environ['TOKENIZERS_PARALLELISM']='false'

perplexity = evaluate.load("perplexity", module_type="metric")

def score_ppl(input_texts: List[str], model_id: str = 'distilgpt2', batch_size: int = 32,):
    """

    uses https://huggingface.co/spaces/evaluate-metric/perplexity

    By default, model will use cuda if available!

    print(list(results.keys()))
    >>>['perplexities', 'mean_perplexity']
    """
    ppl = perplexity.compute(model_id='distilgpt2', add_start_token=True, predictions=input_texts, batch_size=batch_size)
    
    return ppl["mean_perplexity"], np.array(ppl['perplexities']).std()
    

# from tqdm import tqdm
# import math
# import torch


# # os.environ['TRANSFORMERS_CACHE']='.'

# from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# # Load pre-trained model (weights)
# model = GPT2LMHeadModel.from_pretrained('distilgpt2')
# model = model.eval()
# if torch.cuda.is_available():
#     model = model.cuda()
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')

# def distilGPT2_perplexity_score(sentence: str):
#     inputs = tokenizer(sentence, return_tensors='pt')['input_ids'].to(model.device)
#     loss, logits = model(inputs, labels=inputs)[:2]
#     return math.exp(loss)

# def score_ppl(texts: List[str]):
#     return np.array([distilGPT2_perplexity_score(text) for text in tqdm(texts, desc="Scoring PPL", total=len(texts))])

# # GPT doesn't have a padding token so not clear how to do batch scoring of ppls
# # def distilGPT2_batched_perplexity_score(texts: List[str]):
# #     ppls = []
# #     breakpoint()
# #     inputs = tokenizer(texts, return_tensors='pt')['input_ids']
# #     loss, logits = model(inputs.to(model.device), labels=inputs.to(model.device))[:2]
# #     ppls.append(math.exp(loss))
# #     return np.array(ppls)
