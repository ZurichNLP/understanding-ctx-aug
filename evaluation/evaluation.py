#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Optional, Union
import re
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
    from .sentence_processing import count_questions, count_exclamations
    from .reference_metrics import compute_rouge, compute_bleu, compute_meteor, compute_exact_match, compute_self_bleu, compute_chrf
    from .distinct import distinct_n
    from .tokenization import tokenize_texts
    from .novelty import compute_novelty
    from .sentiment import classify_sentiment, classify_sentiment_with_vader
    from .hedge_detection import HedgeHogModel, count_hedge_phrases, hedging_contrast, hedging_management, hedging_evasion
except ImportError:
    from perplexity import score_ppl
    from sentence_processing import count_questions, count_exclamations
    from reference_metrics import compute_rouge, compute_bleu, compute_meteor, compute_exact_match, compute_self_bleu, compute_chrf
    from distinct import distinct_n
    from tokenization import tokenize_texts
    from novelty import compute_novelty
    from sentiment import classify_sentiment, classify_sentiment_with_vader
    from hedge_detection import HedgeHogModel, count_hedge_phrases, hedging_contrast, hedging_management, hedging_evasion

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
    # ap.add_argument('--references_dir', type=str, default=None, help='directory containing reference files')
    ap.add_argument('-o', '--outfile', type=str, default=None, help='')
    ap.add_argument('--output_dir', type=str, default=None, help='')
    ap.add_argument('-v', '--verbose', action='store_true', help='')
    ap.add_argument('--exp_ids', 
                    nargs='*', 
                    type=str, 
                    default=[
                        'baseline', 
                        # 'qu_ctxt_aug1', 
                        'qu_ctxt_aug5', 
                        # 'short_qu_ctxt_aug5',
                        # 'single_qu_ctxt_aug5',
                        # 'xa_dialog', 
                        # 'xa_dialog+qu_ctxt_aug5', 
                        # 'xa_knowledge', 
                        # 'xa_knowledge+qu_ctxt_aug5',
                        'pos_sent_ctxt_aug5', 
                        # 'single_pos_ctxt_aug5',
                        # # 'neu_sent_ctxt_aug5',
                        # 'neg_sent_ctxt_aug5',
                        # 'hedging_contrast_ctxt_aug5',
                        # 'hedging_management_ctxt_aug5',
                        # 'hedging_evasion_ctxt_aug5',
                        # 'ambig_qu_ctxt_aug5',
                        # 'ambig_excl_ctxt_aug5',
                        # 'e_words_ctxt_aug5',
                        # 'd_words_ctxt_aug5',
                        # 'i_words_ctxt_aug5',
                        # 'n_words_ctxt_aug5',
                        # 'excl_ctxt_aug5',
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

    result = {}

    if verbose:
        print('computing string level metrics...')
    result['uniq'] = uniq_response_ratio(sys_outputs)
    lens = np.array([len(text.split()) for text in tokenize_texts(sys_outputs)])
    result['len_mean'] = lens.mean()
    result['len_std'] = lens.std()


    # question count
    if verbose:
        print('counting questions...')
    qc = count_questions(sys_outputs)  
    result['qc_turn_level'] = sum([1 for i in qc if i > 0]) / len(qc)
    result['qc_sent_level'] = qc.sum() / len(qc)

    if verbose:
        print('counting exclamations...')
    ec = count_exclamations(sys_outputs)
    result['ec_turn_level'] = sum([1 for i in ec if i > 0]) / len(ec)
    result['ec_sent_level'] = ec.sum() / len(ec)

    if verbose:
        print('predicting sentiment...')
    # sentiment - we use rule based vader for pos, neg, neu sentiment classification
    sentiment_preds = classify_sentiment_with_vader(sys_outputs)
    result['vader_pos_sents'] = sum([1 for i in sentiment_preds if i['label'] == 'POSITIVE']) / len(sentiment_preds)
    result['vader_neu_sents'] = sum([1 for i in sentiment_preds if i['label'] == 'NEUTRAL']) / len(sentiment_preds)
    result['vader_neg_sents'] = sum([1 for i in sentiment_preds if i['label'] == 'NEGATIVE']) / len(sentiment_preds)

    sentiment_preds = classify_sentiment(sys_outputs, 'distilbert-base-uncased-finetuned-sst-2-english', 128)
    result['pos_sents'] = sum([1 for i in sentiment_preds if i['label'] == 'POSITIVE']) / len(sentiment_preds)
    result['neu_sents'] = sum([1 for i in sentiment_preds if i['label'] == 'NEUTRAL']) / len(sentiment_preds)
    result['neg_sents'] = sum([1 for i in sentiment_preds if i['label'] == 'NEGATIVE']) / len(sentiment_preds)

    # hedging detection
    if verbose:
        print('detecting hedging...')
    result['hedging_contrast'] = count_hedge_phrases(sys_outputs, hedging_contrast) / len(sys_outputs)
    result['hedging_management'] = count_hedge_phrases(sys_outputs, hedging_management) / len(sys_outputs)
    result['hedging_evasion'] = count_hedge_phrases(sys_outputs, hedging_evasion) / len(sys_outputs)

    # uncertainty cues hedging detection
    hedgehog = HedgeHogModel(use_cuda=True, batch_size=128, mp=False)
    predictions, _ = hedgehog.annotate(sys_outputs)
    token_counts, sent_counts = hedgehog.count_hedges(predictions)
    for key, value in sent_counts.items():
        result[f'{key}_sents'] = value / sum(sent_counts.values())
    for key, value in token_counts.items():
        result[f'{key}_tokens'] = value # this is the raw count, not the ratio

    # perplexity
    if verbose:
        print('computing perplexity...')
    ppl_mean, ppl_std = score_ppl(sys_outputs, batch_size=128)
    result['ppl_mean'] = ppl_mean
    result['ppl_std'] = ppl_std

    # distint-n
    if verbose:
        print('computing distint-n...')
    dist = distinct_n(tokenize_texts(sys_outputs)) # returns dict
    result.update(dist)

    # self-bleu
    if verbose:
        print('computing self-bleu...')
    self_bleu = compute_self_bleu(sys_outputs, use_subset=200, is_tokenized=False, verbose=verbose)
    result['self_bleu'] = self_bleu['score']

    return result

def compute_reference_based_metrics(
    sys_outputs: List[str], 
    references: List[List[str]],
    tag: str = '',
    is_tokenized: bool = False,
    verbose: bool = False
    ) -> Dict:
    """
    reference-based metrics (BLEU, ROUGE, METEOR) for KGD

    :tag: 't' for target, 'k' for knowledge, 'd' for dialog
    """
    result = {}
    
    print(f'Computing reference-based metrics for {tag}...')

    bleu = compute_bleu(sys_outputs, references, is_tokenized=is_tokenized, verbose=verbose)
    rouge = compute_rouge(sys_outputs, references, is_tokenized=is_tokenized, verbose=verbose)
    meteor = compute_meteor(sys_outputs, references, is_tokenized=is_tokenized, verbose=verbose)
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
    
    novelty = compute_novelty(sys_outputs, references, is_tokenized=is_tokenized, ignore_case=True, N=4, verbose=verbose)
    chrf = compute_chrf(sys_outputs, references, is_tokenized=is_tokenized, verbose=verbose)

    if tag:
        tag = '_' + tag[0]

    result[f'bleu{tag}'] = bleu['score'] if bleu is not None else None
    result[f'rouge1{tag}'] = rouge['rouge1'] if rouge is not None else None
    result[f'rouge2{tag}'] = rouge['rouge2'] if rouge is not None else None
    result[f'rougeL{tag}'] = rouge['rougeL'] if rouge is not None else None
    result[f'meteor{tag}'] = meteor['meteor'] if meteor is not None else None
    result[f'exact{tag}'] = exact['exact_match'] if exact is not None else None
    result[f'novelty{tag}_1gram'] = novelty.get('1_gram') if novelty is not None else None
    result[f'novelty{tag}_2gram'] = novelty.get('2_gram') if novelty is not None else None
    result[f'novelty{tag}_3gram'] = novelty.get('3_gram') if novelty is not None else None
    result[f'novelty{tag}_4gram'] = novelty.get('4_gram') if novelty is not None else None
    result[f'chrf{tag}'] = chrf.get('score') if chrf is not None else None

    return result    

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
    
    # ('resources', 'models', 'seed_23', 'KGD', 'bart_mini-rndm', 'outputs', 'generations_test_freq_seed\\=0_ml\\=40_lp\\=1.0_ns\\=1_bs\\=4_ds\\=1_temp\\=0.7_tk\\=0_tp\\=0.9.txt')
    model_name_or_path = str(file.parts[4])
    
    if re.search(r'(checkpoint-\d+)', str(file)):
        checkpoint_dir = re.search(r'(checkpoint-\d+)', str(file)).group(1)
    else:
        checkpoint_dir = 'best'

    file_name = file.name
    
    test_file = re.search(r'_(test(_\w+)?)_seed', file_name).group(1) # e.g. 'test_freq', 'test_rare', 'test'

    text_column = 'turns'
    summary_column = 'target'
    knowledge_column = 'knowledge'
    batch_size = 120
    max_length = int(re.search(r'ml=(\d+)', file_name).group(1))
    do_sample = True if re.search(r'ds=(\d)', file_name).group(1) == '1' else False
    top_p = float(re.search(r'tp=(\d+\.\d+)', file_name).group(1))
    top_k = int(re.search(r'tk=(\d+)', file_name).group(1))
    temperature = float(re.search(r'temp=(\d+\.\d+)', file_name).group(1))
    beam_size = int(re.search(r'bs=(\d+)', file_name).group(1))
    num_return_sequences = int(re.search(r'ns=(\d+)', file_name).group(1))
    write_to_file = 'auto' #str(file)
    seed = int(re.search(r'seed=(\d+)_', file_name).group(1))
        
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
    knowledge_snippets: Optional[List[str]] = None,
    dialogs: Optional[List[str]] = None,
    sys_inputs: Optional[List[str]] = None,
    verbose: bool = False
    ):

    start_time = time.time()
    result = {}

    validate_system_outputs(sys_outputs)

    result.update(compute_reference_free_metrics(sys_outputs, verbose=verbose))
    # targets as references (i.e. true references)
    if targets is not None:
        result.update(compute_reference_based_metrics(sys_outputs, targets, 'target', verbose))
    # only knowledge snippets as references
    if knowledge_snippets is not None:
        result.update(compute_reference_based_metrics(sys_outputs, knowledge_snippets, 'knowledge', verbose))
    # only dialogs as references
    if dialogs is not None:
        result.update(compute_reference_based_metrics(sys_outputs, dialogs, 'dialog', verbose))
    # source texts as references
    if sys_inputs is not None: # using model inputs as references
        result.update(compute_reference_based_metrics(sys_outputs, sys_inputs, 'source', verbose))
    elif knowledge_snippets is not None and dialogs is not None:
        combined_inputs = [[' '.join(knowledge_snippets[i] + dialogs[i])] for i in range(len(knowledge_snippets))]
        result.update(compute_reference_based_metrics(sys_outputs, combined_inputs, 'source', verbose))

    end_time = time.time()
    
    print(f'Scored {len(sys_outputs)} system outputs in {end_time-start_time:.2f} seconds.')

    return result


def write_to_csv(result: List[Dict], outfile: Optional[str]):
    df = pd.DataFrame(result)
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
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=1-train_questions-10.txt'))
            elif exp_id == 'qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_questions-10.txt'))
            elif exp_id == 'short_qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-short_questions-5.txt'))
            elif exp_id == 'xa_dialog':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-dialog.txt'))
            elif exp_id == 'xa_dialog+qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-dialog_ctxt=5-train_questions.txt'))
            elif exp_id == 'xa_knowledge':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-knowledge.txt'))
            elif exp_id == 'xa_knowledge+qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_xatt=5-knowledge_ctxt=5-train_questions-10.txt'))
            elif exp_id == 'pos_sent_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-pos_sents-5.txt'))
            elif exp_id == 'neu_sent_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_neu_sents-10.txt'))
            elif exp_id == 'neg_sent_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-neg_sents-5.txt'))
            elif exp_id == 'long_pos_sent_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_pos_sents-10.txt'))
            elif exp_id == 'long_neg_sent_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_neg_sents-10.txt'))
            elif exp_id == 'hedging_contrast_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-hedging_contrast-5.txt'))
            elif exp_id == 'hedging_management_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-hedging_management-5.txt'))
            elif exp_id == 'hedging_evasion_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-hedging_evasion-5.txt'))
            elif exp_id == 'ambig_qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_amibig_questions-10.txt'))
            elif exp_id == 'ambig_excl_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_amibig_exclamations-10.txt'))
            elif exp_id == 'e_words_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-e_words-5.txt'))
            elif exp_id == 'd_words_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-d_words-5.txt'))
            elif exp_id == 'i_words_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-i_words-5.txt'))
            elif exp_id == 'n_words_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-n_words-5.txt'))
            elif exp_id == 'excl_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_exclamations-5.txt'))
            elif exp_id == 'single_qu_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-train_questions-1.txt'))
            elif exp_id == 'single_pos_ctxt_aug5':
                generations_files = sorted(Path(args.generations).glob(f'*tp=0.9_ctxt=5-pos_sents-1.txt'))
            else:
                raise ValueError(f'Unknown experiment id: {exp_id}')

            assert len(generations_files) != 0, f'No generations files found for {exp_id} in {args.generations}'

            references_file = args.references_file
            print(f'Using references from {references_file} ...')

            # filter out generations files that do not match the references file
            # i.e. if the references file is test_freq.json, only evaluate generations files that contain 'freq' in their name
            generations_files = list(filter(lambda f: Path(references_file).stem in f.name, generations_files))
            print(f'Found {len(generations_files)} generations files for {exp_id} ...')
            
            results = []
            for generations_file in generations_files:
                gen_args = parse_args_from_file(generations_file)

                print(f'====== Scoring {generations_file} ======')
                    
                sys_outputs = read_lines(generations_file)
                source = read_json_lines(references_file)
                                
                refs_t = [[i] for i in source['target']]
                refs_d = [[' '.join(i)] for i in source['turns']]
                refs_k = [[i] for i in source['knowledge']] if 'knowledge' in source else None

                scores = score_kgd_generation(
                    sys_outputs=sys_outputs,
                    targets=refs_t,
                    knowledge_snippets=refs_k,
                    dialogs=refs_d,
                    verbose=args.verbose,
                )
                
                result = {**gen_args, **scores}
                # results keys should contain the following columns
                # if set(result.keys()) != set(expected_keys):
                #     print(f'[!] WARNING: results keys do not match expected keys! This may be due to a change in the evaluation script. If you are sure the results are correct, please update the expected_keys variable.')
                #     print(f'\tExpected: {set(expected_keys)}')
                #     print(f'\tFound: {set(result.keys())}')

                print(result)
                results.append(result)

            models_evaluated = [r['model_name_or_path'] for r in results]
            assert len(set(models_evaluated)) == 1, f'Expected one unique model name but found {len(set(models_evaluated))} models in {generations_files}'
            model_evaluated = models_evaluated[0].split('/')[-1]
            outfile = Path(args.output_dir) / f"{model_evaluated}-{exp_id}.csv"
            print(f'Writing all results to {outfile} ...')
            write_to_csv(results, outfile)

    elif Path(args.generations).is_file():
        print(f'====== Scoring {args.generations} ======')
        sys_outputs = read_lines(args.generations)
        source = read_json_lines(args.references_file) if args.references_file is not None else None
        
        refs_t = [[i] for i in source['target']]
        refs_k = [[i] for i in source['knowledge']]
        refs_d = [[' '.join(i)] for i in source['turns']]

        result = score_kgd_generation(
            sys_outputs=sys_outputs,
            targets=refs_t,
            knowledge_snippets=refs_k,
            dialogs=refs_d,
            verbose=args.verbose,
            )

        result['file'] = args.generations

        write_to_csv([result], args.outfile)

if __name__ == '__main__':
    args = set_args()
    main(args)