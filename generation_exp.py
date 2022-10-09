#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import json

from inference import InferenceModel
from evaluation.eval import score_kgd_generation

def set_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", type=str, required=True, default=None, help="path to the finetuned model folder")
    ap.add_argument("--checkpoint", type=str, default=None, help="checkpoint to use for generation, if none, the best (lowest) checkpoint is used")
    ap.add_argument("--dataset", type=str, required=True, default="resources/data/Topical-Chat/KGD/test_freq.json", help="path to the dataset for generation")
    ap.add_argument("--max_predict_samples", type=float, default=1.0, help="maximum number of samples to predict as a percentage of the dataset size")
    ap.add_argument("-o", "--output_dir", type=str, default='results', required=False, help="path to the output directory for evaluated results csvs")
    ap.add_argument("-g", "--generation_dir", type=str, default=None, required=False, help="path to the output directory for generation outputs")
    ap.add_argument("--debug", action="store_true", help="")
    ap.add_argument("-s", "--seed", type=int, nargs="*", default=[0, 42, 983, 8630, 284], help="list of random seeds to use")
    ap.add_argument("--data_seed", type=int, default=0, help="random seed for the dataset split. We keep this fixed for all seeds to ensure that the same samples are used for all seeds")
    ap.add_argument("-b", "--batch_size", type=int, default=120, help="batch size to use for inference. Adjust this depending on the size of the GPU and the model.")
    ap.add_argument(
        "--exp_id", 
        required=False, 
        default='baseline', 
        choices=[
            "baseline",
            "xa_knowledge",
            "xa_dialog",
            "qu_ctxt_aug5",
            "qu_ctxt_aug1",
            "xa_knowledge+qu_ctxt_aug5",
            "xa_dialog+qu_ctxt_aug5",
            "tagged_qu_ctxt_aug5", # for debugging
        ],
        help="experiment id"
        )
    return ap.parse_args()

def get_best_checkpoint(model_dir):
    """
    returns the best checkpoint directory name
    # NOTE: this assumes that the trainer_state.json file is present in the model_dir
    """
    trainer_state = Path(model_dir) / 'trainer_state.json'
    with open(trainer_state, 'r') as f:
        trainer_state = json.load(f)    
    
    best_checkpoint = trainer_state['best_model_checkpoint']
    if best_checkpoint is not None:
        return Path(best_checkpoint).name # return the checkpoint directory name from the path
    else: # assume that the best checkpoint is the earliest checkpoint with save_total_limit = 1
        checkpoints = sorted(list(Path(model_dir).glob('checkpoint-*')))
        return checkpoints[0].stem


topical_chat_data_config = {
    "text_column": "turns",
    "summary_column": "target",
    "knowledge_column": "knowledge",
}

baseline_config = {
    "max_length": 40,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 0,
    "temperature": 0.7,
    "beam_size": 4,
    "num_return_sequences": 1,
    "write_to_file": "auto",
}

debug_config = {
    "max_predict_samples": 5,
    "write_to_file": '',
    "verbose": True,
    "debug": True,
}

experiment_configs = {
    "xa_knowledge": {
        "cross_attention_bias_value": 5,
        "bias_profile": "knowledge",
    },
    "xa_dialog": {
        "cross_attention_bias_value": 5,
        "bias_profile": "dialog",
    },
    "qu_ctxt_aug5": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    },
    # "tagged_qu_ctxt_aug5": {
    #     "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/questions_tagged.txt",
    #     "context_code_attention_bias_value": 5,
    #     "max_context_examples": 10,
    # },
    "qu_ctxt_aug1": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/train_questions.txt",
        "context_code_attention_bias_value": 1,
        "max_context_examples": 10,
    },
}

def print_args(args: Dict):
    string = ''
    for k, v in args.items():
        string += f'--{k}={v} '
    return string

if __name__ == "__main__":
    args = set_args()
    
    checkpoint = get_best_checkpoint(args.model_dir) if args.checkpoint is None else args.checkpoint # fetch checkpoint with the lowest step number if not provided
    
    if not args.debug:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        outfile = Path(args.output_dir) / f'{Path(args.model_dir).name}-{checkpoint.replace("-", "_")}-{args.exp_id}.csv'
        print(f'Writing generation run results to {outfile}')
        if outfile.exists():
            print(f'[!] Overwriting {outfile}')
    
    # set generation args
    gen_args = {
        "model_name_or_path": args.model_dir,
        "checkpoint_dir": checkpoint,
        "batch_size": args.batch_size,
        "test_file": args.dataset, # add dataset passed as argument
        "data_seed": args.data_seed,
        "max_predict_samples": args.max_predict_samples,
        "output_dir": args.generation_dir,
        }

    gen_args.update(topical_chat_data_config)
    
    gen_args.update(baseline_config)
    
    for exp_id in args.exp_id.split('+'):
        gen_args.update(experiment_configs.get(exp_id, {}))
    
    if args.debug:
        gen_args.update(debug_config)
    
    results = []
    for seed in args.seed:
        
        gen_args['seed'] = seed

        # to execute seperate processes from the command line, uncomment this        
        arg_string = print_args(gen_args)
        print(f'Running inference.py with the following args:\n\t{arg_string}')
        # os.system(f'python inference.py {arg_string}')
        
        m = InferenceModel(gen_args)
        predict_dataset = m.load_test_set_for_generation() # default: resources/data/Topical-Chat/KGD/test_freq.json
        outputs = m.generate_KGD(predict_dataset)
        outputs = [o[0] for o in outputs] # take only the first output for each input (in case of multiple return sequences)
        
        if not args.debug:
            scored = score_kgd_generation(
                outputs, 
                targets=[[i] for i in predict_dataset['target']],
                knowledge_snippets=[[i] for i in predict_dataset['knowledge']],
                dialogs=[[' '.join(i)] for i in predict_dataset['turns']],
                verbose=True if args.debug else False,
                )
            
            experiment_result = {**gen_args, **scored}
            results.append(experiment_result)
        
            df = pd.DataFrame(results)    
            df.to_csv(outfile, index=False)
            print(f'Finished generation runs. Results saved to {outfile}')
        else:
            for i, o in enumerate(outputs):
                print(f'{i}: {o}')





