#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import re
from pathlib import Path

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers

logger = logging.getLogger(__name__)

def preprocess_topical_chat_dataset(examples, tokenizer, **kwargs):
    """
    prepare knowledge-grounded dialogue data for training according to 
    the description in Appendix A.1 in https://arxiv.org/abs/2106.06411
    """
    
    # speaker ID tags are expected to look something like `<speaker1>`
    # since these are not part of the model vocab, they get segmented
    # so we accound for the additional tokens this creates by adding 
    # a constant to data_args.history_bucket_size
    speaker_id_tok = tokenizer(kwargs['speaker_id_tag'], add_special_tokens=False, return_length=True)
    speaker_id_tok_len = speaker_id_tok['length'][0]
    # logger.info(
    #         f"tokenized speaker ID tag has length: {speaker_id_tok_len}, e.g. {tokenizer.tokenize(data_args.speaker_id_tag)}"
    #     )

    # remove pairs where at least one record is None
    inputs, targets = [], []
    for i in range(len(examples[kwargs['text_column']])):
        if examples[kwargs['text_column']][i] and examples[kwargs['summary_column']][i] and examples[kwargs['knowledge_column']][i]:
            # tokenize knowledge snippet and pad/truncate to length of history bucket size
            knowledge_tok = tokenizer(
                examples[kwargs['knowledge_column']][i], 
                max_length=kwargs['knowledge_bucket_size'], 
                padding='max_length', 
                truncation=True, 
                add_special_tokens=False)['input_ids']
            
            # tokenize **each** turn and pad/truncate to length of  bucket length
            turns_tok = [tokenizer(
                turn, 
                max_length=kwargs['history_bucket_size'] + speaker_id_tok_len, 
                padding='max_length', 
                truncation=True, 
                add_special_tokens=False)['input_ids'] for turn in examples[kwargs['text_column']][i]]

            turns_tok = [tok_seq for turn_tok in turns_tok for tok_seq in turn_tok] # flatten
            input_tok = knowledge_tok + turns_tok # prepend knowledge sequence to dialogue history
            inputs.append(input_tok)
    
            target_text = examples[kwargs['summary_column']][i]
            if target_text.startswith(kwargs['speaker_id_tag'][:5]):
                target_text = target_text[len(kwargs['speaker_id_tag']):].strip() # remove speaker tag from target

            target_tok = tokenizer(
                target_text, 
                max_length=kwargs['max_target_length'], 
                truncation=True, 
                add_special_tokens=True)['input_ids']
            targets.append(target_tok)

    if kwargs['source_prefix']: # will skip if None or empty string
        prefix_tok = tokenizer(kwargs['source_prefix'], add_special_tokens=False)['input_ids']
        inputs = [prefix_tok + inp_tok for inp_tok in inputs]
    
    # add bos and eos tokens to input sequence
    if tokenizer.bos_token_id is not None: # use bos token if the model has one, otherwise use eos (T5 doesn't have bos)
        inputs = [[tokenizer.bos_token_id] + inp_tok for inp_tok in inputs]
    else:
        inputs = [[tokenizer.eos_token_id] + inp_tok for inp_tok in inputs]
    # else: # if using BERT2BERT, we use the sep token as a stand-in for the eos token
    #     inputs = [[tokenizer.sep_token_id] + inp_tok for inp_tok in inputs]

    model_inputs = tokenizer.prepare_for_model(inputs, add_special_tokens=False)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        # NOTE this is somewhat unneccessary, but it is closer to the original code...
        labels = tokenizer.prepare_for_model(targets, add_special_tokens=False)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if kwargs['pad_to_max_length'] and kwargs['ignore_pad_token_for_loss']:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function(examples, tokenizer, **kwargs):
    """
    Adapted from run_summarization.py
    
    default preprocessing function for summarization datasets
    """
    # remove pairs where at least one record is None
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            if isinstance(examples[text_column][i], list):
                inputs.append(' '.join(examples[text_column][i]))
            else:
                inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_data(model_args, data_args, training_args):
    """    
    Adapted from run_summarization.py

    Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub).
    
    For CSV/JSON files this script will use the first column for the full texts and the second column for the
    summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    
    In distributed training, the load_dataset function guarantee that only one local process can concurrently
    download the dataset.
    See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    https://huggingface.co/docs/datasets/loading_datasets.html.
    """
    
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    return raw_datasets

def prepare_data_for_model(model_args, data_args, training_args, tokenizer, logger):
    """
    Adapted from run_summarization.py
    """
    train_dataset, eval_dataset, test_dataset = None, None, None
    
    raw_datasets = load_data(model_args, data_args, training_args)
    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return train_dataset, eval_dataset, test_dataset

    # Get the column names for input/target.
    dataset_columns = None
    
    if data_args.text_column is None:
        # expected at position 1
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # For KDG with topical-chat
    if data_args.knowledge_column is not None:
        knowledge_column = data_args.knowledge_column
        if knowledge_column not in column_names:
            raise ValueError(
                f"--knowledge_column' value '{data_args.knowledge_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            data_args.max_train_samples = (
                int(len(train_dataset) * data_args.max_train_samples) 
                if data_args.max_train_samples <= 1.0 
                else int(data_args.max_train_samples)
            )
            logger.info(f"Using {data_args.max_train_samples} samples for training!")
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            if data_args.knowledge_column is not None:
                train_dataset = train_dataset.map(
                    preprocess_topical_chat_dataset,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                    fn_kwargs={
                        'tokenizer': tokenizer,
                        **data_args.__dict__,
                    },
                )
            else: # original
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            data_args.max_eval_samples = (
                int(len(eval_dataset) * data_args.max_eval_samples) 
                if data_args.max_eval_samples <= 1.0 
                else int(data_args.max_eval_samples)
            )
            logger.info(f"Using {data_args.max_eval_samples} samples for evaluation!") 
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            if data_args.knowledge_column is not None:
                eval_dataset = eval_dataset.map(
                    preprocess_topical_chat_dataset,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                    fn_kwargs={
                        'tokenizer': tokenizer,
                        **data_args.__dict__,
                    },
                )
            else: # original
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            data_args.max_predict_samples = (
                int(len(predict_dataset) * data_args.max_predict_samples) 
                if data_args.max_predict_samples <= 1.0 
                else int(data_args.max_predict_samples)
            )
            logger.info(f"Using {data_args.max_predict_samples} samples for testing!") 
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            if data_args.knowledge_column is not None:
                predict_dataset = predict_dataset.map(
                    preprocess_topical_chat_dataset,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                    fn_kwargs={
                        'tokenizer': tokenizer,
                        **data_args.__dict__,
                    },
                )
            else: # original
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

    if training_args.do_train and data_args.persist_datasets:
        train_dataset.to_json(Path(training_args.output_dir) / "train_dataset.json")
        eval_dataset.to_json(Path(training_args.output_dir) / "eval_dataset.json")
        predict_dataset.to_json(Path(training_args.output_dir) / "predict_dataset.json")
        logger.info(f"Saved datasets to {training_args.output_dir}")

    return train_dataset, eval_dataset, predict_dataset

if __name__ == "__main__":
    pass
