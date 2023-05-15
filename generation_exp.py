#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
from typing import List, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import json

from inference import InferenceModel
from constants import *
from evaluation.evaluation import score_kgd_generation

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, default=None, help="path to the finetuned model folder")
    ap.add_argument("--checkpoint", type=str, default=None, help="checkpoint to use for generation, if required")
    ap.add_argument("--dataset", type=str, required=False, default="kgd", help="dataset type")
    ap.add_argument("--test_file", type=str, required=False, default="resources/data/Topical-Chat/KGD/test_freq.json", help="path to the dataset for generation")
    ap.add_argument("--max_predict_samples", type=float, default=1.0, help="maximum number of samples to predict as a percentage of the dataset size")
    ap.add_argument("--output_dir", type=str, default='results', required=False, help="path to the output directory for evaluated results csvs")
    ap.add_argument("--generation_dir", type=str, default=None, required=False, help="path to the output directory for generation outputs")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 42, 983, 8630, 284], help="list of random seeds to use for generation")
    ap.add_argument("--data_seed", type=int, default=0, help="random seed for the dataset split. We keep this fixed for all seeds to ensure that the same samples are used for all seeds")
    ap.add_argument("--batch_size", type=int, default=120, help="batch size to use for inference. Adjust this depending on the size of the GPU and the model.")
    ap.add_argument("--greedy", action="store_true", help="whether or not to use greedy decoding")
    ap.add_argument("--exp_id", required=False, default='baseline', help="experiment id")
    ap.add_argument("--debug", action="store_true", help="set for test runs")

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

def print_args(args: Dict):
    string = ''
    for k, v in args.items():
        string += f'--{k}={v} '
    return string

if __name__ == "__main__":
    major_start = time.time() # time experiment run
    args = set_args()
    
    checkpoint = args.checkpoint if args.checkpoint is not None else 'best'

    if not args.debug:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        outfile = Path(args.output_dir) / f'{Path(args.model_dir).name}-{checkpoint.replace("-", "_")}-{args.exp_id}.csv'
        print(f'Writing generation run results to {outfile}')
        if outfile.exists():
            print(f'[!] Overwriting {outfile}')
    else:
        outfile = None
    
    # set generation args
    gen_args = {
        "model_name_or_path": args.model_dir,
        "checkpoint_dir": args.checkpoint,
        "batch_size": args.batch_size,
        "test_file": args.test_file, # add dataset passed as argument
        "data_seed": args.data_seed,
        "max_predict_samples": args.max_predict_samples,
        "output_dir": args.generation_dir,
        }

    # dataset-specific args
    if args.dataset.lower() in ['kgd', 'topical_chat']:
        gen_args.update(TOPICAL_CHAT_DATA_CONFIG)
    elif args.dataset.lower() in ['csd', 'cs_dialog']:
        gen_args.update(COMMONSENSE_DIALOG_DATA_CONFIG)
    elif args.dataset.lower() in ['dd', 'daily_dialog']:
        gen_args.update(DAILY_DIALOG_DATA_CONFIG)

    # basic decoding args
    if args.greedy:
        gen_args.update(GREEDY_CONFIG)        
    else:
        gen_args.update(BASELINE_CONFIG)
    
    # experiment-specific args
    # note: it's possible to pass multiple experiment ids separated by '+', e.g. --exp_id=xa_knowledge+qu_ctxt_aug5
    for exp_id in args.exp_id.split('+'):
        if args.dataset.lower() in ['kgd', 'topical_chat']:
            exp_config = KGD_EXPERIMENT_CONFIGS.get(exp_id, None)
        elif args.dataset.lower() in ['csd', 'cs_dialog']:
            exp_config = CSD_EXPERIMENT_CONFIGS.get(exp_id, None)
            gen_args.update({'beam_size': 1}) # reduce beam size for CD    
        elif args.dataset.lower() in ['dd', 'daily_dialog']:
            exp_config = DD_EXPERIMENT_CONFIGS.get(exp_id, None)
            
        if exp_id != 'baseline':
            if exp_config is not None:
                gen_args.update(exp_config)
            else:
                raise ValueError(f'Invalid experiment id: {exp_id}')
    
    # debug args for test runs
    if args.debug:
        gen_args.update(DEBUG_CONFIG)
    
    results = []
    seed_count = 0
    for seed in args.seeds:
        seed_count += 1
        minor_start = time.time() # time generation run

        print('\n***')
        print(f'Running generation with seed {seed} ({seed_count}/{len(args.seeds)})')
        gen_args['seed'] = seed

        # to execute seperate processes from the command line, uncomment this        
        arg_string = print_args(gen_args)
        print(f'Running inference.py with the following args:\n\t{arg_string}')
        # os.system(f'python inference.py {arg_string}')
        
        m = InferenceModel(gen_args)
        predict_dataset = m.load_test_set_for_generation() # default: resources/data/Topical-Chat/KGD/test_freq.json
        
        outputs = m.generate_KGD(predict_dataset)
        outputs = [o[0] for o in outputs] # take only the first output for each input (in case of multiple return sequences)
        
        minor_end = time.time()
        print(f'◴◵◶◷ Finished generation run with seed {seed} in {minor_end - minor_start:.2f} seconds ◴◵◶◷')

        if args.debug:
            for i, o in enumerate(outputs):
                print(f'{i}: {o}')
            sys.exit()
        
        else:

            minor_start = time.time() # time scoring run

            if args.dataset.lower() in ['kgd', 'topical_chat']:
                scored = score_kgd_generation(
                    outputs, 
                    targets=[[i] for i in predict_dataset['target']],
                    knowledge_snippets=[[i] for i in predict_dataset['knowledge']],
                    dialogs=[[' '.join(i)] for i in predict_dataset['turns']],
                    verbose=True if args.debug else False,
                    )
            elif args.dataset.lower() in ['csd', 'cs_dialog']:
                scored = score_kgd_generation(
                    outputs, 
                    targets=[[i] for i in predict_dataset['target']],
                    knowledge_snippets=[[i] for i in predict_dataset['context']],
                    dialogs=[[' '.join(i)] for i in predict_dataset['turns']],
                    verbose=True if args.debug else False,
                    )
            
            elif args.dataset.lower() in ['dd', 'daily_dialog']:
                scored = score_kgd_generation(
                    outputs, 
                    targets=[[i] for i in predict_dataset['target']],
                    knowledge_snippets=None, #[[''] for i in predict_dataset['target']],
                    dialogs=[[' '.join(i)] for i in predict_dataset['turns']],
                    verbose=True if args.debug else False,
                    )
                
            experiment_result = {**gen_args, **scored}
            results.append(experiment_result)
        
            # note: we save the results after each generation run, 
            # overwriting the results files each time,
            # to avoid losing data if the process is interrupted
            df = pd.DataFrame(results)
            print(f'Results currently has shape {df.shape}. Saving to {outfile} ...')
            # print('-'*70)
            # print(f'\t{results}')
            # print('-'*70)
            df.to_csv(outfile, index=False)
            
            minor_end = time.time()
            print(f'◴◵◶◷ Finished scoring in {minor_end - minor_start:.2f} seconds ◴◵◶◷')
            
    major_end = time.time()
    print(f'◴◵◶◷ Finished all generation runs in {major_end - major_start:.2f} seconds ◴◵◶◷')



