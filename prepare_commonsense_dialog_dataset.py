#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Union, Optional
from pprint import pprint
from dataclasses import dataclass, asdict
import argparse
import random
from tqdm import tqdm
import re
from transformers import AutoTokenizer


@dataclass
class dialogue_instance:
    turns: List[str]
    context: str
    target: str


class CommonSenseDialogDataset:
    
    def __init__(self, data_dir: str, split: str, verbose: bool = False):
        
        self.split = split
        self.data_dir = data_dir
        self.seed = 42 # for reproducibility
        self.verbose = verbose

        self.annotated_dialogues = self._load_json(Path(data_dir) / f'{split}.json')
                
    def _load_json(self, file: Union[Path, str]) -> Dict:
        with open(file) as f:
            return json.load(f)

    def extract_dialogue(self, dialogue_id: str, history_length: int = 5, verbose: bool = False) -> List:
        """
        extract all contextualised source-target sequence pairs from a given dialogue.
        """
        # breakpoint()        
        dialogue = self.annotated_dialogues[dialogue_id]['turns']
        context = self.annotated_dialogues[dialogue_id]['context']

        # if len(dialogue) != 6:
        #     print(f'Warning: dialogue {dialogue_id} has {len(dialogue)} turns, expected 6!')

        src_tgt_pairs = []
        current_dialogue = []
        
        for i, turn in enumerate(dialogue):    
            if i % 2 == 0:
                speaker_id = 1
            else:
                speaker_id = 2
                
            current_dialogue.append(f"<speaker{speaker_id}> {turn}")

            if len(current_dialogue) > history_length:

                di = dialogue_instance(
                    turns = self.normalize_whitespace(current_dialogue[:-1]), 
                    context = self.normalize_whitespace(context), 
                    target = self.normalize_whitespace(current_dialogue[-1])
                    )

                src_tgt_pairs.append(di)
                current_dialogue.pop(0)

            # add padding to the end of the dialogue if it's shorter than history_length
            elif i == len(dialogue) - 1: # dialogue is over before history_length turns
                # breakpoint()
                # fill up the dialogue with placeholders
                while len(current_dialogue) < history_length + 1:
                    # speaker_id = 1 if current_dialogue[0].startswith('<speaker2>') else 2
                    # current_dialogue.insert(0, f"<speaker{speaker_id}>")
                    current_dialogue.insert(0, f"")

                di = dialogue_instance(
                    turns = self.normalize_whitespace(current_dialogue[:-1]), 
                    context = self.normalize_whitespace(context), 
                    target = self.normalize_whitespace(current_dialogue[-1])
                    )

                src_tgt_pairs.append(di)
                
        return src_tgt_pairs

    def get_all_dialogues(self, history_length: int = 5, verbose: bool = False) -> List:
        all_dialogues = []
        for dialogue_id in self.annotated_dialogues.keys():
            all_dialogues.extend(self.extract_dialogue(dialogue_id, history_length=history_length, verbose=verbose))
        
        print(f'Extracted {len(all_dialogues)} contextualised grounded dialogues!')
            
        return all_dialogues

    def write_to_file(self, dialogues: List, save_dir: str, shuffle: bool = False, seed: int = 42) -> None:

        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)

        output_file = Path(save_dir) / f'{self.split}.json' # files are jsonl format

        if shuffle:
            random.seed(seed)
            random.shuffle(dialogues)

        with open(output_file, 'w', encoding='utf8') as f:
            c = 0
            for dialogue in (dialogues):
                c += 1
                f.write(json.dumps(asdict(dialogue), ensure_ascii=False) + '\n')
            print(f'Wrote {c} dialogues to {output_file}')

    
    @staticmethod
    def normalize_whitespace(text: Union[List, str]) -> Union[List, str]:
    
        def clean_string(string: str) -> str:
            string = re.sub(r'\n', ' ', string)
            string = re.sub(r'\s+', ' ', string)
            return string.strip()

        if isinstance(text, list):
            return [clean_string(string) for string in text]
        else:
            return clean_string(text)
        
    @staticmethod
    def tokenize_dialogues(
        dialogues: List, 
        tokenizer: str, 
        history_bucket_size: int, 
        context_bucket_size: int, 
        split: str, 
        save_dir: str, 
        verbose: bool = False
        ) -> List:

        """
        Tokenize all dialogues in a list of dialogue instances.
        """
        
        outpath = Path(save_dir) / f'{tokenizer.replace("/", "-")}' 
        
        if not outpath.exists():
            outpath.mkdir(parents=True)

        src_output_file = outpath / f'{split}.src'
        tgt_output_file = outpath / f'{split}.tgt'
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        speaker_id = re.compile(r'<speaker([0-9])>\s?') # remove speaker tag from target
        
        with open(src_output_file, 'w', encoding='utf8') as src_file:
            with open(tgt_output_file, 'w', encoding='utf8') as tgt_file:
            
                for dialogue in tqdm(dialogues, total=len(dialogues), desc='Tokenizing dialogues', disable=verbose):
                
                    context_text = ' '.join(tokenizer.tokenize(dialogue.context, max_length=context_bucket_size, padding='max_length', truncation=True))
                    # '<speaker1>' and '<speaker2>' (e.g. '<', 'spe', 'aker', '1', '>') tags are split into 4 tokens with BART's tokenizer, 
                    # so we account for this by extending the max length of each history turn
                    history = ' '.join([' '.join(tokenizer.tokenize(turn, max_length=history_bucket_size+4, padding='max_length', truncation=True)) for turn in dialogue.turns])

                    src_tok = tokenizer.bos_token + ' ' + context_text + ' ' + history # + ' ' + tokenizer.eos_token
                    tgt_tok = tokenizer.bos_token + ' ' + ' '.join(tokenizer.tokenize(re.sub(speaker_id, '', dialogue.target), max_length=256, truncation=True))
            
                    src_file.write(src_tok + '\n')
                    tgt_file.write(tgt_tok + '\n')

        return

def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--data_dir', type=str, required=True, help='path to data directory')
    ap.add_argument('--split', type=str, default='train', help='train, valid, or test')
    ap.add_argument('--history_length', type=int, default=5, help='number of turns that make up the source sequence dialogue history')
    ap.add_argument('--context_length', type=int, default=1, help='number of context snippets to use for dialogue grounding')
    
    ap.add_argument('--tokenizer', type=str, default='facebook/bart-base', help='name or path to model tokenizer')
    ap.add_argument('--context_bucket_size', type=int, default=32, help='number of tokens in bucket for context snippets')
    ap.add_argument('--history_bucket_size', type=int, default=25, help='number of tokens in bucket for a historical turn')
    
    ap.add_argument('--verbose', action='store_true', help='print out debug information')
    ap.add_argument('--save_dir', type=str, default='KGD', help='path to save directory')
    
    return ap.parse_args()


if __name__ == "__main__":

    args = set_args()

    cd = CommonSenseDialogDataset(args.data_dir, args.split, args.verbose)

    dialogues = cd.get_all_dialogues()
    # breakpoint()
    cd.write_to_file(dialogues, args.save_dir)
    
    # # tokenize inputs according to description in the paper
    # tc.tokenize_dialogues(dialogues, tokenizer=args.tokenizer, 
    #     history_bucket_size=args.history_bucket_size, 
    #     context_bucket_size=args.context_bucket_size,
    #     split=args.split, 
    #     save_dir=args.save_dir, 
    #     verbose=args.verbose)
    
