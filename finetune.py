#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuning the library models for sequence to sequence.

Adapted from the examples scripts in HuggingFace's transformers library.
"""

import logging
import os
import json
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
import re
from pathlib import Path
from functools import partial

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EncoderDecoderModel,
    PreTrainedTokenizer,
    EvalPrediction,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_callback import EarlyStoppingCallback

from evaluation.eval import score_kgd_generation
from data import prepare_data_for_model
from hf_args import ModelArguments, DataTrainingArguments

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def parse_args() -> Tuple[Dict]:
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def model_init(model_args, data_args, training_args, tokenizer, logger):
    """
    Load pretrained model following recommendation: 
    https://discuss.huggingface.co/t/fixing-the-random-seed-in-the-trainer-does-not-produce-the-same-results-across-runs/3442
    """
    if model_args.is_encoder_decoder:
        if '+' in model_args.model_name_or_path:
            encoder_name, decoder_name = model_args.model_name_or_path.split('+')
            logger.info(f"Loading {encoder_name}-{decoder_name} as encoder-decoder")
            raise NotImplementedError("We don't support encoder-decoder models with two different encoders and decoders yet.")
            # For this to work we need to do some magic to utilize both tokenizers...
            # e.g. https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
            # model.config.vocab_size = model.config.decoder.vocab_size
        else:
            encoder_name = decoder_name = model_args.model_name_or_path

        model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name, tie_encoder_decoder=model_args.tie_encoder_decoder)
        logger.info(f"Loaded {encoder_name}-{decoder_name} as encoder-decoder. Tied encoder-decoder: {model_args.tie_encoder_decoder}")
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id        
        if tokenizer.bos_token_id is None: # BERT doesn't have a bos_token_id, so repurpose cls_token_id
            model.config.bos_token_id = tokenizer.bos_token_id = tokenizer.cls_token_id
        if tokenizer.eos_token_id is None: # BERT doesn't have a eos_token_id, so repurpose sep_token_id
            model.config.eos_token_id = tokenizer.eos_token_id = tokenizer.sep_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(len(tokenizer))

        # config params needed for generation
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.max_length = 128
        model.config.min_length = 2
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 1.0
        model.config.num_beams = 4

    else:

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        model.resize_token_embeddings(len(tokenizer))
    
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


    logger.info(f'Total model parameters: {model.num_parameters()}')
    logger.info(f'Total trainable model parameters: {model.num_parameters(only_trainable=True)}')
    logger.info(f'Total trainable model parameters excluding embeddings: {model.num_parameters(only_trainable=True, exclude_embeddings=True)}')

    return model

def postprocess_text(texts: List[str]) -> List[str]:
    # texts = [text.strip() for text in texts]
    # rougeLSum expects newline after each sentence
    texts = ["\n".join(nltk.sent_tokenize(text.strip())) for text in texts]
    return texts

def make_compute_metrics(model_args, data_args, training_args, tokenizer, logger) -> Callable[[EvalPrediction], Dict]:
    """
    wraps the custom compute_metrics function with the data_args and tokenizer as arguments (https://github.com/huggingface/transformers/issues/9264)
    """
    def compute_metrics(eval_preds, model_args=model_args, data_args=data_args, training_args=training_args, tokenizer=tokenizer, logger=logger) -> Dict:
        """
        Scores KGD with evaluation functions in evalutate/eval.py. 
        Note, this computes a number of metrics, so can be slow...
        """
        preds = eval_preds.predictions # np.array of shape (num_examples, max_seq_len)
        labels = eval_preds.label_ids # np.array of shape (num_examples, max_seq_len)
        inputs = eval_preds.inputs # np.array of shape (num_examples, max_seq_len)
        
        # TODO: custom metrics for validations
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True) if inputs is not None else None

        # Some simple post-processing
        # decoded_preds = postprocess_text(decoded_preds)
        # decoded_labels = postprocess_text(decoded_labels)
        # decoded_inputs = postprocess_text(decoded_inputs)
        
        result = score_kgd_generation(
            sys_outputs=decoded_preds, 
            targets=[[l] for l in decoded_labels], # expects references to be a list of list of strings
            sys_inputs=[[i] for i in decoded_inputs] if decoded_inputs is not None else None # expects references to be a list of list of strings
            )
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items() if v is not None}
        
        # save outputs in output_dir
        if data_args.write_intermediate_eval_results:
            i = len(list(Path(training_args.output_dir).glob('eval_predictions*')))
            outfile = Path(training_args.output_dir) / f'eval_predictions.{i}.json'
            with open(outfile, 'w', encoding='utf8') as outf:
                json_entry = {
                    'inputs': decoded_inputs,
                    'labels': decoded_labels,
                    'preds': decoded_preds,
                    'metrics': result
                }
                outf.write(f'{json.dumps(json_entry)}\n')
                logger.info(f'Wrote eval predictions and metrics to {outfile}')

        return result
    return compute_metrics

def main():

    model_args, data_args, training_args = parse_args()
    
    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(
            project=data_args.project_name, 
            tags=[f'train_samples_{data_args.max_train_samples}', data_args.train_file, data_args.validation_file, data_args.test_file, model_args.model_name_or_path],
            )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    if training_args.fp16 and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/t5-small-lm-adapt",
    ]:
        # update training_args.fp16 with T5 models, since this seems unstable
        training_args.fp16 = False
        logger.warning(f"fp16 is unstable with T5 models, so we're turning it off! training_args.fp16 = {training_args.fp16}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    train_dataset, eval_dataset, predict_dataset = prepare_data_for_model(model_args, data_args, training_args, tokenizer, logger)
    
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    training_callbacks = None
    if data_args.early_stopping:
        training_callbacks = [EarlyStoppingCallback(data_args.early_stopping_patience, data_args.early_stopping_threshold)]
        logging.info(f"early stopping callback set with patience {data_args.early_stopping_patience} \
            and threshold {data_args.early_stopping_threshold}")
    
    # setup compute metrics function which is passed to the Trainer
    compute_metrics = make_compute_metrics(model_args, data_args, training_args, tokenizer, logger)

    # derive the number of valiation runs at equal intervals per epoch (added for experimentation)
    if data_args.eval_runs_per_epoch > 1:
        steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        training_args.eval_steps = steps_per_epoch // data_args.eval_runs_per_epoch
        logger.info(f"Updated `eval_steps` to {training_args.eval_steps} for {data_args.eval_runs_per_epoch} runs per epoch. \
            Validation set has {len(eval_dataset)} samples with batch size {training_args.per_device_eval_batch_size}.")

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model_init(model_args, data_args, training_args, tokenizer, logger),
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=training_callbacks,
    )

    # Initial evaluation (before any fine-tunining)
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    # Pre-training Evaluation
    if training_args.do_eval and training_args.evaluation_strategy == "steps":
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)
        print(metrics)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Post-training Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
