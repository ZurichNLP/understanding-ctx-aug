#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import torch
from multiprocessing import Pool, cpu_count
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import random
import re
from torch.multiprocessing import Pool, Process, set_start_method
import ray
import pandas
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
Reads in target utterances from jsonl file, classifies them with a sentiment analysis model, and writes the results to a tsv file.

# vader
python evaluation/sentiment.py \
    --input_file resources/data/Topical-Chat/KGD/test_freq.json \
    --target_column "target" \
    --output_file resources/data/Topical-Chat/KGD/sentiment/test_freq_vader.tsv


python evaluation/sentiment.py \
    --input_file resources/data/Topical-Chat/KGD/test_freq.json \
    --target_column "target" \
    --batch_size 256 \
    --model_name "distilbert-base-uncased-finetuned-sst-2-english" \
    --output_file resources/data/Topical-Chat/KGD/sentiment/test_freq_vader.tsv

Parts of this script are adapted from https://towardsdatascience.com/parallel-inference-of-huggingface-transformers-on-cpus-4487c28abe23
"""

PIPE = None
set_start_method("spawn", force=True)
random.seed(42)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--target_column", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--n_cores", type=int, default=1)
    args = parser.parse_args()
    return args

def init_pipe(model_name, batch_size):
    """
    This will load the pipeline on demand on the current PROCESS/THREAD and load it only once.
    """
    global PIPE
    if PIPE is None:
        # NOTE:  All pipelines can use batching. However, this is not automatically a win for performance. 
        # It can be either a 10x speedup or 5x slowdown depending on hardware, data and the actual model being used.
        # Batching is only recommended on GPU. If you are using CPU, don’t batch.
        device = 0 if torch.cuda.is_available() else 'cpu'
        if device != 'cpu':
            print(f'Running on cuda:{device}')
        PIPE = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, batch_size=batch_size, device=device)
    return PIPE


def classify_sentiment_with_ray(texts, n_cores, model_name):
    """
    For parallel inference on CPUs, we use Ray.
    """    
    print('Number of CPUs:', n_cores)
    
    # Start Ray cluster
    ray.init(num_cpus=n_cores, ignore_reinit_error=True)
    pipe = init_pipe(model_name)

    print(ray.cluster_resources())
    
    """
    The command ray.put(x) would be run by a worker process or by the driver process (the driver process is the one running your script). 
    It takes a Python object and copies it to the local object store (here local means on the same node). 
    Once the object has been stored in the object store, its value cannot be changed.
    In addition, ray.put(x) returns an object ID, which is essentially an ID that can be used to refer to the newly created remote object. 
    If we save the object ID in a variable with x_id = ray.put(x), 
    then we can pass x_id into remote functions, 
    and those remote functions will operate on the corresponding remote object.
    """
    pipe_id = ray.put(pipe)

    # @ray.remote decorator enables to use this function in distributed setting
    @ray.remote
    def predict(pipeline, texts):
        # print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        return pipeline(texts)

    # Predict multipe texts on all available CPUs and time the inference duration    
    # Run the function using multiple cores and gather the results
    predictions = ray.get([predict.remote(pipe_id, text.lower()) for text in texts])

    # Stop running Ray cluster
    ray.shutdown()
    
    return predictions
    
def get_texts(input_file, target_column, sample):
    if input_file is not None:
        extension = input_file.split(".")[-1]
        if extension in ['json', 'csv']: 
            if target_column is None:
                raise RuntimeError('target column must be specified for json and csv files')
            else: 
                dataset_dict = load_dataset(extension, data_files={'test': args.input_file})
                assert target_column in dataset_dict['test'].column_names
                texts = dataset_dict['test'][target_column]
        elif extension == 'txt':
            with open(input_file, 'r', encoding='utf8') as f:
                texts = [line.strip() for line in f.readlines()]
    
    # strip speaker info
    texts = [re.sub('(<speaker1>|<speaker2>)\s+', '', text) for text in texts]
    texts = [re.sub('\n', ' ', text).strip() for text in texts]

    if sample > 0: # for debugging
        texts = random.sample(texts, sample)
        print(f'[!] sampled {sample} texts')
    
    return texts

def classify_sentiment(texts, model_name, batch_size):
    pipe = init_pipe(model_name, batch_size)
    tokenizer_kwargs = {'padding':True, 'truncation':True, 'max_length':512,}
    results = [pred for pred in tqdm(pipe(texts, **tokenizer_kwargs), total=len(texts))]
    return results

def parse_vader_result(d):
    """
    Uses recommended threshold values from VADER (https://github.com/cjhutto/vaderSentiment#about-the-scoring)
    """
    if d['compound'] >= 0.05:
        return {'label': 'POSITIVE', 'score': d['compound']}
    elif d['compound'] <= -0.05:
        return {'label': 'NEGATIVE', 'score': d['compound']}
    else:
        return {'label': 'NEUTRAL', 'score': d['compound']}

def classify_sentiment_with_vader(texts):
    sid = SentimentIntensityAnalyzer()
    results = [parse_vader_result(sid.polarity_scores(text)) for text in texts]
    return results

if __name__ == "__main__":
    args = set_args()

    texts = get_texts(args.input_file, args.target_column, args.sample)
    
    start = time.time()

    if args.model_name:
        sentiment_engine = 'hf'
        if args.n_cores > 1:
            predictions = classify_sentiment_with_ray(texts, args.n_cores, args.model_name)
        else:
            predictions = classify_sentiment(texts, args.model_name, args.batch_size)
    
    else: # use vader
        sentiment_engine = 'vader'
        predictions = classify_sentiment_with_vader(texts)
    
    end = time.time()
    print(f'Prediction time: {end-start:.2f} seconds')

    if args.output_file is not None:
        c = 0
        with open(args.output_file, 'w', encoding='utf8') as outf:
            for text, prediction in zip(texts, predictions):
                outf.write(f"{text}\t{prediction['label']}\t{prediction['score']}\n")
                c += 1
        print(f'wrote {c} lines to {args.output_file}')
    else:
        for text, prediction in zip(texts, predictions):
            print(f"{text}\t{prediction['label']}\t{prediction['score']}")
