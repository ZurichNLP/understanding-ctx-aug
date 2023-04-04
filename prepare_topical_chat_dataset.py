
"""
# # Prepare Topical-Chat Dataset for Knowlegde-Grounded Dialogue Model
# 
# "We use the setting introduced in the Topical-Chat dataset (Gopalakrishnan et al., 2019) which includes dialogs between two Mechanical Turk workers (a.k.a. Turkers). 
# 
# Based on the previous work (Hedayatnia et al., 2020), we choose the setting where for each turn in the dialog, the knowledge snippet that is the most similar to the ground truth response is selected using TF-IDF and is provided as additional input."
# 
# From Hazarika et al., 2021 (A.1):
# 
# "input comprises a knowledge snippet k and the dialog history h. 
# Here, dialog history is the last five turns in the dialog, with respect to the response. 
# To prepare the input, we assign a fixed number of tokens for each section in the input. 
# 
# We call each section a bucket. If the actual number of tokens of an input section is less 
# than the total tokens assigned for that bucket, we pad the input to infill the empty tokens. 
# In particular, we provide 32 tokens for the knowledge snippet k and 25 tokens for each turn in the dialog history. 
# 
# We start the input sequence with the special token 〈s〉, followed by the knowledge snippet’s bucket.
# Next, we include the dialog history, whose turns use alternate start symbols: 〈speaker1〉, 〈speaker2〉. 
# Overall, our input comprises 163 tokens, 33 knowledge tokens plus 26 turn tokens for each of the 5 turns. 
# On the decoder side, for teacher-forcing, we provide the human response as the input, along with the start token 〈s〉"

NOTE: due to different handling of <speaker1> and <speaker2> tags, our input lengths may differ slightly.

Example Usage:
    python prepare_topical_chat_dataset.py --data_dir resources/data/Topical-Chat --split test_freq


"""

# %%
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

# %%

@dataclass
class dialog_instance:
    turns: List[str]
    knowledge: str
    target: str
    
class TopicalChat:
        
    def __init__(self, data_dir: str, split: str):
        
        self.split = split
        self.data_dir = data_dir
        self.seed = 42 # for reproducibility

        # don't need this because we use the `enriched version`
        # conv_file = Path(data_dir) / f'conversations/{split}.json'
        # self.conv_data = self._load_json(conv_file)

        dialogs_with_linked_knowledge = Path(data_dir) / f'TopicalChatEnriched/{split}.json'
        self.annotated_dialogs = self._load_json(dialogs_with_linked_knowledge)

        reading_set = Path(data_dir) / f'reading_sets/post-build/{split}.json'
        self.knowledge_data = self._load_json(reading_set)
        
        assert len(self.knowledge_data.keys()) == len(self.annotated_dialogs.keys())
        
        # keep track of items that we could not retrieve a knowledge source for
        self.failed = set()
        
    def _load_json(self, file: Union[Path, str]) -> Dict:
        with open(file) as f:
            return json.load(f)
        
    @staticmethod
    def extract_knowledge_segment(knowledge: Dict, turn_ks: Dict, agent: str, verbose: bool = False) -> str:
        """
        extract the appropriate knowledge segment given an knowledge entry provided to a 
        conversation agent and a pointer dictionary with linking information. The pointer
        dictionary comes from the Enriched Topical-Chat dataset [Hedayatnia et al., 2020 
        https://aclanthology.org/2020.inlg-1.46.pdf]

        Args:

            :knowledge: 

                The knowledge data provided to an agent in Topical-Chat.

                This dictionary obj should contain the following structure:

                    {
                        “FS1”: {
                         “entity”: <entity name>,
                         “shortened_wiki_lead_section”: <section text>,
                         “fun_facts”: [ <fact1_text>, <fact2_text>,…]
                            },
                        “FS2”:…
                                    },
                        ....
                    },

            :turn_ks: 

                Pointers to specific knowledge segment for a given turn in Topical-Chat.

                This dictionary obj should contain the following structure:

                    {"score": 0.67, "ds": "fun_facts", "section": "FS1", "index": 0}
                    Note: index can also be start_idx and end_idx for string objects


        """

        if verbose:
            print(f'\nSECTION {section}')
            print(turn_ks)

        knowledge_text = None
        
        if turn_ks['ds'] == 'article':
            knowledge_text = knowledge['article']
            try:
                knowledge_text = knowledge_text[turn_ks['section']]
                knowledge_text = knowledge_text[turn_ks['start_index']:turn_ks['end_index']]
                if verbose:
                    print(f"ARTICLE KNOWLEDGE from {turn_ks['start_index']}:{turn_ks['end_index']}: {knowledge_text}")
            except:
                if verbose:
                    print(f"[!] Failed to retrieve data source {turn_ks['section']} in {knowledge_text}")
                return None
            
        elif turn_ks['ds'] == 'wiki': # shortened_wiki_lead_section / summarized_wiki_lead_section
            section = turn_ks['section']
            knowledge_text = knowledge[agent][section]
            try:
                knowledge_text = knowledge_text['shortened_wiki_lead_section'] # shortened occurs far more frequently, so try this first
            except KeyError:
                knowledge_text = knowledge_text['summarized_wiki_lead_section']
            knowledge_text = knowledge_text[turn_ks['start_index']:turn_ks['end_index']]

            if verbose:
                print(f"WIKI KNOWLEDGE from {turn_ks['start_index']}:{turn_ks['end_index']}: {knowledge_text}")

        elif turn_ks['ds'] == 'fun_facts':
            section = turn_ks['section']
            knowledge_text = knowledge[agent][section]['fun_facts'][turn_ks['index']]
            if verbose:
                print(f"FUN FACT at INDEX {turn_ks['index']}: {knowledge_text}")

        else:
            raise NotImplementedError(f'[!] Cannot parse {turn_ks}')

        return knowledge_text


    def extract_knowledge_grounded_dialog(self, dialog_id: str, history_length: int = 5, verbose: bool = False) -> List:
        """
        extract all knowledge-grounded source-target sequence pairs from a given dialog.
        """
        
        anno_dialog = self.annotated_dialogs[dialog_id]
        knowledge = self.knowledge_data[dialog_id]

        src_tgt_pairs = []
        current_dialog = []
        
        for i, turn in enumerate(anno_dialog['content']):    

            current_dialog.append(f"<speaker{turn['agent'][-1]}> {' '.join(turn['message'])}")

            if len(current_dialog) > history_length:

                knowledge_text = self.extract_knowledge_segment(knowledge, turn['gt_turn_ks'], turn['agent'], verbose=verbose) # if isinstance(knowledge, dict) else None

                if not knowledge_text:
                    self.failed.add(knowledge['article'])

                di = dialog_instance(
                    turns = self.normalize_whitespace(current_dialog[:-1]), 
                    knowledge = self.normalize_whitespace(knowledge_text), 
                    target = self.normalize_whitespace(current_dialog[-1])
                    )

                src_tgt_pairs.append(di)
                current_dialog.pop(0)

        return src_tgt_pairs

    def get_all_dialogs(self, history_length: int = 5, verbose: bool = False) -> List:
        all_dialogs = []
        for dialog_id in self.annotated_dialogs.keys():
            all_dialogs.extend(self.extract_knowledge_grounded_dialog(dialog_id, history_length=history_length, verbose=verbose))
        
        if len(self.failed):
            print(f"[!] failed to locate knowledge data for the following items:")
            for i in self.failed:
                print(i)
        else:
            print(f'Extracted {len(all_dialogs)} knowledge grounded dialogs!')
            
        return all_dialogs

    def write_to_file(self, dialogs: List, save_dir: str, shuffle: bool = False, seed: int = 42) -> None:

        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)

        output_file = Path(save_dir) / f'{self.split}.json' # files are jsonl format

        if shuffle:
            random.seed(seed)
            random.shuffle(dialogs)

        with open(output_file, 'w', encoding='utf8') as f:
            c = 0
            for dialog in (dialogs):
                c += 1
                f.write(json.dumps(asdict(dialog), ensure_ascii=False) + '\n')
            print(f'Wrote {c} dialogs to {output_file}')

    
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
    def tokenize_dialogs(
        dialogs: List, 
        tokenizer: str, 
        history_bucket_size: int, 
        knowledge_bucket_size: int, 
        split: str, 
        save_dir: str, 
        verbose: bool = False
        ) -> List:

        """
        Tokenize all dialogs in a list of dialog instances.
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
            
                for dialog in tqdm(dialogs, total=len(dialogs), desc='Tokenizing dialogs', disable=verbose):
                
                    knowledge_text = ' '.join(tokenizer.tokenize(dialog.knowledge, max_length=knowledge_bucket_size, padding='max_length', truncation=True))
                    # '<speaker1>' and '<speaker2>' (e.g. '<', 'spe', 'aker', '1', '>') tags are split into 4 tokens with BART's tokenizer, 
                    # so we account for this by extending the max length of each history turn
                    history = ' '.join([' '.join(tokenizer.tokenize(turn, max_length=history_bucket_size+4, padding='max_length', truncation=True)) for turn in dialog.turns])

                    src_tok = tokenizer.bos_token + ' ' + knowledge_text + ' ' + history # + ' ' + tokenizer.eos_token
                    tgt_tok = tokenizer.bos_token + ' ' + ' '.join(tokenizer.tokenize(re.sub(speaker_id, '', dialog.target), max_length=256, truncation=True))
            
                    src_file.write(src_tok + '\n')
                    tgt_file.write(tgt_tok + '\n')

        return
#%%

def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--data_dir', type=str, required=True, help='path to data directory')
    ap.add_argument('--split', type=str, default='train', help='train, valid_freq, or test_freq')
    ap.add_argument('--history_length', type=int, default=5, help='number of turns that make up the source sequence dialog history')
    ap.add_argument('--knowledge_length', type=int, default=1, help='number of knowledge snippets to use for dialog grounding')
    
    ap.add_argument('--tokenizer', type=str, default='facebook/bart-base', help='name or path to model tokenizer')
    ap.add_argument('--knowledge_bucket_size', type=int, default=32, help='number of tokens in bucket for knowledge snippets')
    ap.add_argument('--history_bucket_size', type=int, default=25, help='number of tokens in bucket for a historical turn')
    
    ap.add_argument('--verbose', action='store_true', help='print out debug information')
    ap.add_argument('--save_dir', type=str, default='KGD', help='path to save directory')
    
    return ap.parse_args()

#%% 
if __name__ == "__main__":

    # tc = TopicalChat(data_dir, 'test_freq')
    # tc = TopicalChat(data_dir, 'test_rare')
    # test_id = 't_d004c097-424d-45d4-8f91-833d85c2da31'
    # tc.extract_knowledge_grounded_dialog(test_id)

    args = set_args()

    tc = TopicalChat(args.data_dir, args.split)

    dialogs = tc.get_all_dialogs()
    tc.write_to_file(dialogs, args.save_dir)
    
    # # tokenize inputs according to description in the paper
    # tc.tokenize_dialogs(dialogs, tokenizer=args.tokenizer, 
    #     history_bucket_size=args.history_bucket_size, 
    #     knowledge_bucket_size=args.knowledge_bucket_size,
    #     split=args.split, 
    #     save_dir=args.save_dir, 
    #     verbose=args.verbose)
    
    

