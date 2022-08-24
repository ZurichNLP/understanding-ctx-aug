#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Dict, Optional, Union
from pathlib import Path
import pandas as pd

from inference import InferenceModel
from evaluation.eval import score_kgd_generation

def set_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", type=str, required=True, help="path to the finetuned model folder")
    ap.add_argument("-o", "--out_dir", type=str, default='results', required=False, help="path to the output directory")
    ap.add_argument("-d", "--debug", action="store_true", help="")
    ap.add_argument("-s", "--seed", type=int, nargs="*", default=[0, 42, 983, 8630, 284], help="list of random seeds to use")
    ap.add_argument(
        "--exp_id", 
        required=False, 
        default='baseline', 
        choices=[
            "baseline",
            "xa_knowledge",
            "xa_dialog",
            "qu_ctxt_aug",
            "xa_knowledge+qu_ctxt_aug",
            "xa_dialog+qu_ctxt_aug",
        ],
        help="experiment id"
        )
    return ap.parse_args()

def get_best_checkpoint(model_dir):
    """
    returns the checkpoint directory name with the lowest step number
    """
    checkpoints = sorted(list(Path(model_dir).glob('checkpoint-*')))
    return checkpoints[0].stem


topical_chat_data_config = {
    "test_file": "resources/data/Topical-Chat/KGD/test_freq.json",
    "text_column": "turns",
    "summary_column": "target",
    "knowledge_column": "knowledge",
}

baseline_config = {
    "batch_size": 120,
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
    "qu_ctxt_aug": {
        "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/questions.txt",
        "context_code_attention_bias_value": 5,
        "max_context_examples": 10,
    }
}

def print_args(args: Dict):
    string = ''
    for k, v in args.items():
        string += f'--{k}={v} '
    return string

if __name__ == "__main__":
    args = set_args()
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    outfile = Path(args.out_dir) / f'{Path(args.model_dir).stem}-{args.exp_id}.csv'
    if outfile.exists() and not args.debug:
        print(f'[!] Overwriting {outfile}')

    checkpoint = get_best_checkpoint(args.model_dir)
    model_args = {
        "model_name_or_path": args.model_dir,
        "checkpoint_dir": checkpoint,
        }
    
    gen_args = model_args
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
        # arg_string = print_args(gen_args)
        # print(f'python inference.py {arg_string}')
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
        else:
            for i, o in enumerate(outputs):
                print(f'{i}: {o}')

        





