#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Optional, Union
import json
import argparse
from pathlib import Path
import time
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

try:
    from .perplexity import score_ppl
    from .sentence_processing import count_questions
    from .reference_metrics import compute_rouge, compute_bleu, compute_meteor, compute_exact_match
    from .distinct import distinct_n
    from .tokenization import tokenize_texts
    from .novelty import compute_novelty
except ImportError:
    from perplexity import score_ppl
    from sentence_processing import count_questions
    from reference_metrics import compute_rouge, compute_bleu, compute_meteor, compute_exact_match
    from distinct import distinct_n
    from tokenization import tokenize_texts
    from novelty import compute_novelty

expected_keys = [
    'model_name_or_path', 'checkpoint_dir', 'test_file', 'text_column', 'summary_column', 
    'knowledge_column', 'batch_size', 'max_length', 'do_sample', 'top_p', 'top_k', 'temperature', 
    'beam_size', 'num_return_sequences', 'write_to_file', 'seed', 'uniq', 'qc_turn_level', 
    'qc_sent_level', 'ppl_mean', 'ppl_std', 'intra_dist1', 'intra_dist2', 'inter_dist1', 
    'inter_dist2', 'bleu_t', 'rouge1_t', 'meteor_t', 'bleu_k', 'rouge1_k', 'meteor_k', 
    'bleu_d', 'rouge1_d', 'meteor_d'
    ]

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('generations', type=str, help='file or diectory of files containing generated outputs from inference.py')
    ap.add_argument('-r', '--references_file', type=str, default=None, help='e.g. `resources/data/Topical-Chat/KGD/test_freq.json`')
    ap.add_argument('-o', '--outfile', type=str, default=None, help='')
    ap.add_argument('--output_dir', type=str, default=None, help='')
    ap.add_argument('-v', '--verbose', action='store_true', help='')
    ap.add_argument('--exp_ids', 
                    nargs='*', 
                    type=str, 
                    default=[
                        'baseline', 
                        'qu_ctxt_aug1', 
                        'qu_ctxt_aug5', 
                        'xa_dialog', 
                        'xa_dialog+qu_ctxt_aug5', 
                        'xa_knowledge', 
                        'xa_knowledge+qu_ctxt_aug5'
                        ],
                    help='experiment ids to run evaluation for (by default, will evaluate outputs for all)')

    return ap.parse_args()

def read_lines(file: str, sep: str = '\t'):
    lines = []
    with open(file, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            line = line.split(sep)[0]
            lines.append(line.strip())
    return lines

def reshape_data(data: List[Dict]):
    """
    
    """
    reshaped = {}
    keys = list(data[0].keys())
    for key in keys:
        reshaped[key] = []
        for line in data:
            reshaped[key].append(line[key])
    return reshaped

def read_json_lines(file: str):
    lines = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(json.loads(line.strip()))
    return reshape_data(lines)
    
def uniq_response_ratio(texts: List[str]):
    return len(set(texts)) / len(texts)

def compute_reference_free_metrics(
    sys_outputs: List[str], 
    verbose: bool = False
    ) -> Dict:
    """
    reference-free metrics
    """

    results = {}

    results['uniq'] = uniq_response_ratio(sys_outputs)
    
    # question count
    qc = count_questions(sys_outputs)  
    results['qc_turn_level'] = sum([1 for i in qc if i > 0]) / len(qc)
    results['qc_sent_level'] = qc.sum() / len(qc)
    
    # perplexity
    ppl_mean, ppl_std = score_ppl(sys_outputs, batch_size=128)
    results['ppl_mean'] = ppl_mean
    results['ppl_std'] = ppl_std

    # distint-n
    dist = distinct_n(tokenize_texts(sys_outputs)) # returns dict
    results.update(dist)

    return results

def compute_reference_based_metrics(
    sys_outputs: List[str], 
    references: List[List[str]],
    tag: str = '',
    verbose: bool = False
    ) -> Dict:

    """
    reference-based metrics (BLEU, ROUGE, METEOR) for KGD

    :tag: 't' for target, 'k' for knowledge, 'd' for dialog
    """
    results = {}

    if verbose:
        print(f'Computing reference-based metrics for {tag}...')

    bleu = compute_bleu(sys_outputs, references, is_tokenized=False, verbose=verbose)
    rouge = compute_rouge(sys_outputs, references, is_tokenized=False, verbose=verbose)
    meteor = compute_meteor(sys_outputs, references, is_tokenized=False, verbose=verbose)
    exact = compute_exact_match(
        sys_outputs, 
        [' '.join(r) for r in references], 
        is_tokenized=False, 
        regexes_to_ignore = [r'<speaker1>\s*', r'<speaker2>\s*'],
        ignore_case = False,
        ignore_punctuation = False,
        ignore_numbers = False,
        verbose=verbose
        )
    novelty = compute_novelty(sys_outputs, references, is_tokenized=False, ignore_case=True, N=4, verbose=verbose)

    if tag:
        tag = '_' + tag[0]

    results[f'bleu{tag}'] = bleu['score'] if bleu is not None else None
    results[f'rouge1{tag}'] = rouge['rouge1'] if rouge is not None else None
    results[f'meteor{tag}'] = meteor['meteor'] if meteor is not None else None
    results[f'exact{tag}'] = exact['exact_match'] if exact is not None else None
    results[f'novelty{tag}_1gram'] = novelty.get('1_gram') if novelty is not None else None
    results[f'novelty{tag}_2gram'] = novelty.get('2_gram') if novelty is not None else None
    results[f'novelty{tag}_3gram'] = novelty.get('3_gram') if novelty is not None else None
    results[f'novelty{tag}_4gram'] = novelty.get('4_gram') if novelty is not None else None

    return results    

def validate_system_outputs(sys_outputs: List[str]) -> List[str]:
    """
    check if system outputs are valid
    """
    problematic = []
    for i in range(len(sys_outputs)):
        if len(sys_outputs[i].strip().split()) < 1:
            sys_outputs[i] = 'n/a.'
            problematic.append(i+1) # offset by 1 to match line number in file
    if len(problematic) > 0:
        print(f'[!] {len(problematic)} problematic system outputs: Check the following lines: {problematic}')
    return sys_outputs    

def parse_args_from_file(file: Path) -> Dict:
    """
    hack to parse args from a generation file for post-hoc evaluation
    """
    # breakpoint()
    model_name_or_path = str(file.parents[2])
    checkpoint_dir = str(file.parents[1].name)
    
    file_args = file.stem.split('_')
    # ['generationstest', 'freq', 'seed=0', 'ml=40', 'lp=1.0', 'ns=1', 'bs=4', 'ds=1', 'temp=0.7', 'tk=0', 'tp=0.9']
    test_file = file_args[0][11:]+'_'+file_args[1]+'.json'
    text_column = 'turns'
    summary_column = 'target'
    knowledge_column = 'knowledge'
    batch_size = 120
    max_length = int(file_args[3].split('=')[1])
    do_sample = True if file_args[7].split('=')[1] == '1' else False
    top_p = float(file_args[10].split('=')[1])
    top_k = int(file_args[9].split('=')[1])
    temperature = float(file_args[8].split('=')[1])
    beam_size = int(file_args[7].split('=')[1])
    num_return_sequences = int(file_args[6].split('=')[1])
    write_to_file = 'auto' #str(file)
    seed = int(file_args[2].split('=')[1])
        
    return {
        'model_name_or_path': model_name_or_path,
        'checkpoint_dir': checkpoint_dir,
        'test_file': test_file,
        'text_column': text_column,
        'summary_column': summary_column,
        'knowledge_column': knowledge_column,
        'batch_size': batch_size,
        'max_length': max_length,
        'do_sample': do_sample,
        'top_p': top_p,
        'top_k': top_k,
        'temperature': temperature,
        'beam_size': beam_size,
        'num_return_sequences': num_return_sequences,
        'write_to_file': write_to_file,
        'seed': seed
    }

def score_kgd_generation(
    sys_outputs: List[str], 
    targets: Optional[List[str]],
    knowledge_snippets: Optional[List[str]],
    dialogs: Optional[List[str]],
    verbose: bool = False
    ):

    start_time = time.time()
    results = {}

    validate_system_outputs(sys_outputs)

    results.update(compute_reference_free_metrics(sys_outputs, verbose=verbose))
    
    if targets is not None:
        results.update(compute_reference_based_metrics(sys_outputs, targets, 'target', verbose))
    if knowledge_snippets is not None:
        results.update(compute_reference_based_metrics(sys_outputs, knowledge_snippets, 'knowledge', verbose))
    if dialogs is not None:
        results.update(compute_reference_based_metrics(sys_outputs, dialogs, 'dialog', verbose))
    
    end_time = time.time()
    
    print(f'Scored {len(sys_outputs)} system outputs in {end_time-start_time:.2f} seconds.')

    return results


def write_to_csv(results: List[Dict], outfile: Optional[str]):
    df = pd.DataFrame(results)
    if not outfile:
        print(df.to_csv(index=False))
    else:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outfile, index=False)
        print(f'Wrote {len(df)} results to {outfile}')
    return

def main(args):
    if Path(args.generations).is_dir(): # run evaluation for each file in the directory
        # this is a bit hacky and only intended for post-hoc evalautions of pipeline runs...
        for exp_id in args.exp_ids:
            if exp_id == 'baseline':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9.txt'))
            elif exp_id == 'qu_ctxt_aug1':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=1-*questions-10.txt'))
            elif exp_id == 'qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-*questions-10.txt'))
            elif exp_id == 'xa_dialog':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-dialog.txt'))
            elif exp_id == 'xa_dialog+qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-dialog_ctxt=5-*questions-10.txt'))
            elif exp_id == 'xa_knowledge':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-knowledge.txt'))
            elif exp_id == 'xa_knowledge+qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-knowledge_ctxt=5-*questions-10.txt'))
            
            assert len(generations_files) != 0, f'No generations files found for {exp_id} in {args.generations}'

            results = []
            for generations_file in generations_files:
                print(f'====== Scoring {generations_file} ======')
                sys_outputs = read_lines(generations_file)
                source = read_json_lines(args.references_file) if args.references_file is not None else None
                
                gen_args = parse_args_from_file(generations_file)
                
                refs_t = [[i] for i in source['target']]
                refs_k = [[i] for i in source['knowledge']]
                refs_d = [[' '.join(i)] for i in source['turns']]

                scores = score_kgd_generation(
                    sys_outputs=sys_outputs,
                    targets=refs_t,
                    knowledge_snippets=refs_k,
                    dialogs=refs_d,
                    verbose=args.verbose,
                )
                
                result = {**gen_args, **scores}
                
                # results keys should contain the following columns
                if set(result.keys()) != set(expected_keys):
                    print(f'[!] WARNING: results keys do not match expected keys: {set(result.keys())} vs {set(expected_keys)}')
                    print(f'This may be due to a change in the evaluation script. If you are sure the results are correct, please update the expected_keys variable.')
                    
                results.append(result)

            models_evaluated = [r['model_name_or_path'] for r in results]
            assert len(set(models_evaluated)) == 1, f'Expected 1 model name but found {len(set(models_evaluated))} models in {generations_files}'
            model_evaluated = models_evaluated[0].split('/')[-1]
            outfile = Path(args.output_dir) / f"{model_evaluated}-{exp_id}.csv"
            print(f'Writing results to {outfile} ...')
            write_to_csv(results, outfile)

    elif Path(args.generations).is_file():
        print(f'====== Scoring {args.generations} ======')
        sys_outputs = read_lines(args.generations)
        source = read_json_lines(args.references_file) if args.references_file is not None else None
        
        refs_t = [[i] for i in source['target']]
        refs_k = [[i] for i in source['knowledge']]
        refs_d = [[' '.join(i)] for i in source['turns']]

        results = score_kgd_generation(
            sys_outputs=sys_outputs,
            targets=refs_t,
            knowledge_snippets=refs_k,
            dialogs=refs_d,
            verbose=args.verbose,
            )

        results['file'] = args.generations

        write_to_csv([results], args.outfile)

if __name__ == '__main__':
    args = set_args()
    main(args)