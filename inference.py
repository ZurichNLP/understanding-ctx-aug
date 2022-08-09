import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re
from pathlib import Path
import random

from tqdm import tqdm
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import torch
import transformers
from transformers import (
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
)

from train import tokenize_function, DataTrainingArguments, ModelArguments

# Setup logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = 10 # training_args.get_process_log_level()
logger.setLevel(log_level)
# datasets.utils.logging.set_verbosity(log_level)
# transformers.utils.logging.set_verbosity(log_level)
# transformers.utils.logging.enable_default_handler()
# transformers.utils.logging.enable_explicit_format()

@dataclass
class InferenceArguments:
    """
    Arguments pertaining to running generation/inference with pre-trained/fine-tuned model.
    """

    checkpoint_dir: str = field(
        default=None,
        metadata={"help": "Path to fine-tuned model checkpoint"}
    )
    
    output_dir: str = field(
        default=None,
        metadata={"help": "Path to output directory"}
    )

    seed: int = field(
        default=42,
        metadata={"help": "random seed"}
    )

    use_cuda: bool = field(
        default=True,
        metadata={"help": "Use GPU if available"}
    )

    batch_size: int = field(
        default=3,
        metadata={"help": "Batch size for predictions"}
    )

    min_length: int = field(
        default=None,
        metadata={"help": "Minimum length of generated text"}
    )

    max_length: int = field(
        default=64,
        metadata={"help": "Maximum length of generated text"}
    )

    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for generated text"}
    )

    no_early_stop: bool = field(
        default=False,
        metadata={"help": "Disable early stopping on generate"}
    )

    num_return_sequences: int = field(
        default=1,
        metadata={"help": "Number of sequences to generate"}
    )

    beam_size: int = field(
        default=4,
        metadata={"help": "Number of beams for beam search"}
    )

    do_sample: bool = field(
        default=False,
        metadata={"help": "Sample instead of greedy decoding"}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for generation"}
    )
    
    top_k: int = field(
        default=0,
        metadata={"help": "Number of top k tokens to keep for top-k sampling"}
    )

    top_p: float = field(
        default=0.0,
        metadata={"help": "Probability of top-p sampling"}
    )

    cross_attention_bias: int = field(
        default=1,
        metadata={"help": "Value used to bias cross attention. Default = 1, i.e. no bias. "
            "A bias of 0 acts similarly to setting a custom attention mask for the cross attention."}
    )

    cross_attention_type: str = field(
        default='uniform',
        metadata={"help": ""}
    )

    context_augmentation_examples: str = field(
        default='',
        metadata={"help": "source for context examples if using context augmentation as described by Hazarika et al., 2021. "
        "If a file path is provided, example sentences are expected to be one per line"}
    )

    max_context_examples: int = field(
        default=10,
        metadata={"help": "number of context examples to use for context augmentation"}
    )

    context_code_attention_bias: int = field(
        default=1,
        metadata={"help": "Value used to bias cross attention given context augmentation. Default = 1, i.e. no bias. "
            "A bias of 0 acts similarly to setting a custom attention mask for the cross attention."}
    )

    cross_attention_mode: str = field(
        default='',
        metadata={"help": "Where to apply the cross attention bias\n"
            "'' means no cross attention bias (i.e. default)\n"
            "`knowledge` means apply the bias to the knowledge bucket of the input (for use with KGD).\n"
            "`positional` means apply the bias to the position-specific tokens\n"
            }
    )

    cross_attention_positions: str = field(
        default=None,
        metadata={"help": "Start and end positions of the cross attention biad value."}
    )

    write_to_file: str = field(
        default='auto',
        metadata={"help": "Output file for generated text or `auto` to generate outfile name based on generation parameters"}
    )

class InferenceModel:

    def __init__(self, arg_parser: HfArgumentParser):
        
        self.gen_args, self.model_args, self.data_args = arg_parser.parse_args_into_dataclasses()

        # # loading the model you previously trained
        self.model_path = self.model_args.model_name_or_path
        if self.gen_args.checkpoint_dir is not None:
            self.model_path = str(Path(self.model_args.model_name_or_path) / self.gen_args.checkpoint_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = self.model.eval()
        if torch.cuda.is_available() and self.gen_args.use_cuda:
            self.model = self.model.cuda()


    def load_test_set_for_generation(self, dataset: Optional[str] = None):
        ######################################################################
        ## Load test dataset and run pre-processing (e.g. tokenization as required for KDG model)

        if dataset is not None:
            extension = dataset.split(".")[-1]
            dataset_dict = load_dataset(extension, data_files={'test': dataset})

        elif self.data_args.test_file is not None:
            extension = self.data_args.test_file.split(".")[-1]
            dataset_dict = load_dataset(
                extension, 
                data_files={'test': self.data_args.test_file},
                cache_dir=self.model_args.cache_dir,
                )
            
        predict_dataset = dataset_dict['test']

        if self.data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), self.data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

        predict_dataset = predict_dataset.map(
            tokenize_function, # currently defined in train.py
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
            fn_kwargs={
                'tokenizer': self.tokenizer,
                **self.data_args.__dict__,
            },
        )
        
        
        predict_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
        
        return predict_dataset

    def batch_for_generation(self, examples, batch_size: int, context_code: Optional[torch.Tensor] = None):
        
        current_batch = {
            'input_ids': [], 
            'attention_mask': [], 
            'labels': [],
            'turns': [],
            'knowledge': [],
            'target': [],
            'cross_attention_bias': [],
            'context_code': [],
        }

        for i, example in enumerate(examples):
            current_batch['input_ids'].append(example['input_ids'])
            current_batch['attention_mask'].append(example['attention_mask'])
            current_batch['labels'].append(example['labels'])
            current_batch['turns'].append(example.get('turns'))
            current_batch['knowledge'].append(example.get('knowledge'))
            current_batch['target'].append(example.get('target'))

            # get cross attention biases for each individual example (would be fast to do for a batch if all items are the same)
            if self.gen_args.cross_attention_bias != 1:
                current_batch['cross_attention_bias'].append(self.construct_cross_attention_bias(example['attention_mask']))
            if context_code is not None:
                current_batch['context_code'].append(context_code)

            if len(current_batch['input_ids']) == batch_size or i == len(examples) - 1:
                
                current_batch['input_ids'] = torch.stack(current_batch['input_ids']).to(self.model.device)
                current_batch['attention_mask'] = torch.stack(current_batch['attention_mask']).to(self.model.device)
                # TODO: pad to max length before stacking to return true tensor
                # current_batch['labels'] = torch.stack(current_batch['labels']).to(model.device)
                
                # stack cross attention biases for each individual example
                current_batch['cross_attention_bias'] = torch.stack(current_batch['cross_attention_bias']).to(self.model.device) if len(current_batch['cross_attention_bias']) else None
                # to make sure dimensions are correct, we need to ensure the attention 
                # bias vector as the same length as the encoder hidden states which differs depending on the context code
                current_batch['cross_attention_bias'] = self.expand_attention_bias_to_context_code(current_batch['cross_attention_bias'], context_code)
                # stack context code for each individual example
                current_batch['context_code'] = torch.stack(current_batch['context_code']) if len(current_batch['context_code']) else None

                yield current_batch
            
                # reset lists for next batch
                current_batch['input_ids'], current_batch['attention_mask'], current_batch['labels'] = [], [], []
                current_batch['turns'], current_batch['knowledge'], current_batch['target'] = [], [], []
                current_batch['cross_attention_bias'] = []
                current_batch['context_code'] = []
            
    def generate_KGD(
        self, 
        predict_dataset: datasets.Dataset, 
        context_examples: Optional[List[str]] = None,
        ):
        """
        
        """
        if self.gen_args.context_augmentation_examples is not None:
            context_examples = self.load_context_examples(
                self.gen_args.context_augmentation_examples, 
                self.gen_args.max_context_examples, 
                self.gen_args.seed
                )

        context_code = self.encode_context(context_examples) if context_examples else None

        if context_code is not None:
            logger.info(f"Using context code for generation")
        
        outputs = []
        src_seqs = []
        # breakpoint()
        for pred_batch in tqdm(self.batch_for_generation(predict_dataset, self.gen_args.batch_size, context_code=context_code), total=len(predict_dataset) // self.gen_args.batch_size):
            
            model_outputs = self.model.generate(
                pred_batch['input_ids'], 
                attention_mask=pred_batch['attention_mask'],
                max_new_tokens=self.gen_args.max_length, 
                min_length=self.gen_args.min_length,
                num_beams=self.gen_args.beam_size,
                num_return_sequences=self.gen_args.num_return_sequences, 
                early_stopping=not self.gen_args.no_early_stop,
                do_sample=self.gen_args.do_sample, 
                temperature=self.gen_args.temperature, 
                top_k=self.gen_args.top_k, 
                top_p=self.gen_args.top_p,
                decoder_kwargs={'cross_attention_bias': pred_batch['cross_attention_bias'], 'context_code': pred_batch['context_code']},
                )

            batch_outputs = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
            outputs.extend(batch_outputs)
            src_seqs.extend(self.tokenizer.batch_decode(pred_batch['input_ids']))
        # pack outputs into a list of lists, i.e. batch_len x num_return_seqs
        outputs = [outputs[i:i+self.gen_args.num_return_sequences] for i in range(0, len(outputs), self.gen_args.num_return_sequences)]

        # if not decdoing the full test set, assume debug run and print to stdout
        if self.data_args.max_predict_samples is not None:
            for src, gen in zip(src_seqs, outputs):
                print()
                print(src, '\t', ' \t'.join(gen))

        elif self.gen_args.write_to_file:
            outfile = self.get_outfile_name()
            self.write_outputs_to_outfile(outputs, outfile)
        
            
        return outputs

    def get_outfile_name(self):
        """
        Get the name of the output file to write to.
        """
        if self.gen_args.output_dir is not None:
            output_dir = Path(self.gen_args.output_dir)
        else:
            output_dir = Path(self.model_path) / 'outputs'

        if self.gen_args.write_to_file == 'auto':
            outfile = 'generations'
            outfile += f'{Path(self.data_args.test_file).stem}'
            outfile += f'_seed={self.gen_args.seed}'
            outfile += f'_ml={self.gen_args.max_length}'
            outfile += f'_lp={self.gen_args.length_penalty}'
            outfile += f'_ns={self.gen_args.num_return_sequences}'
            outfile += f'_bs={self.gen_args.beam_size}'
            outfile += f'_ds={int(self.gen_args.do_sample)}'
            outfile += f'_temp={self.gen_args.temperature}'
            outfile += f'_tk={self.gen_args.top_k}'
            outfile += f'_tp={self.gen_args.top_p}'
            if self.gen_args.cross_attention_bias != 1: # default value = baseline
                outfile += f'_xatt={self.gen_args.cross_attention_bias}-{self.gen_args.cross_attention_type}-{self.gen_args.cross_attention_mode}'
            if self.gen_args.context_augmentation_examples: # default no context augmentation = baseline
                outfile += f'_ctxt={self.gen_args.context_code_attention_bias}-{Path(self.gen_args.context_augmentation_examples).stem}-{self.gen_args.max_context_examples}'
            outfile += '.txt'
            logger.info(f"Inferred outfile name as {outfile}")
        else:
            outfile = self.gen_args.write_to_file
        return output_dir / outfile

    @staticmethod
    def write_outputs_to_outfile(outputs: List[List[str]], outfile: str):
        """
        For handling cases when num_return_sequences > 1, outputs are expected to be 
        lists of lists of strings with each sub list corresponding to one input sentence.
        Output lines contain seq1\tseq2\tseq3\t...\tseqn
        """
        
        if Path(outfile).exists():
            logger.info(f"Output file {outfile} already exists. It will be overwritten.")
        else:
            Path(outfile).parent.mkdir(parents=True, exist_ok=True)

        with open(outfile, 'w', encoding='utf8') as f:
            c = 0
            for batch_o in outputs:
                for o in batch_o:
                    f.write(o.replace('\t', '  ') + '\t')
                f.write('\n')
                c += 1
        
        logger.info(f"Wrote {c} lines to {outfile}")
        return

    def encode_context(self, context_examples: List[str]) -> torch.Tensor:
        """
        Context augmentation as described by Hazarika et al. (2021):
            given a list of example context sentences, encode them and 
            return a tensor of shape [num_examples, max_length, hidden_size]
        """
        if context_examples is None:
            raise RuntimeError(f"Expected list of context example to encode but found {context_code}")
        context_inputs = self.tokenizer(context_examples, padding=True, return_tensors='pt')
        # get context code
        context_code = self.model.get_encoder()(context_inputs['input_ids'].to(self.model.device), return_dict=True)['last_hidden_state'].mean(dim=0)#.unsqueeze(dim=0)
        logger.info(f'encoded {len(context_examples)} context examples')
        return context_code

    def construct_cross_attention_bias(self, base_tensor: torch.Tensor) -> torch.Tensor:
        """
        Builds the cross attention bias tensor that is passed to the decoder.
        """
        cross_attention_bias = base_tensor.clone()
        if self.gen_args.cross_attention_bias == 1:
            return cross_attention_bias
        else:
            if not self.gen_args.cross_attention_mode:
                return None

            elif self.gen_args.cross_attention_mode == 'knowledge':
                if len(cross_attention_bias.size()) == 1: # single example
                    cross_attention_bias[1:self.data_args.knowledge_bucket_size+1] = self.gen_args.cross_attention_bias
                else:
                    # handle a batch of examples (note, each example receives the same bias)
                    cross_attention_bias[:, 1:self.data_args.knowledge_bucket_size+1] = self.gen_args.cross_attention_bias
            # TODO: arbitrary cross attention biasing
            # elif self.gen_args.cross_attention_mode == 'positional':
            #     if not self.gen_args.cross_attention_bias_positions:
            #         raise RuntimeError(f"Expected cross attention bias positions but found {self.gen_args.cross_attention_bias_positions}")
            #     for start_position, end_position in self.gen_args.cross_attention_bias_positions:
            #         if len(cross_attention_bias.size()) == 1: # single example
            #             cross_attention_bias[start_position:end_position] = self.gen_args.cross_attention_bias
            #         else:
            #             # handle a batch of examples (note, each example receives the same bias)
            #             cross_attention_bias[:, start_position:end_position] = self.gen_args.cross_attention_bias
            else:
                raise RuntimeError(f"Unknown cross attention mode {self.gen_args.cross_attention_mode}")
                
        return cross_attention_bias

    def expand_attention_bias_to_context_code(
        self, 
        cross_attention_bias: Optional[torch.Tensor] = None, 
        context_code: Optional[torch.Tensor] = None):
        """
        Expands the attention bias vector (if provided) to the size of the context code (if provided)
        """
        if cross_attention_bias is not None and context_code is not None:
            context_code_attention_bias = torch.ones([1, context_code.size()[0]], dtype=int, device=context_code.device) * self.gen_args.context_code_attention_bias
            context_code_attention_bias = context_code_attention_bias.repeat(cross_attention_bias.size()[0], 1) # repeat for each example in batch
            cross_attention_bias = torch.cat([context_code_attention_bias, cross_attention_bias], dim=-1)
            return cross_attention_bias
        else:
            return cross_attention_bias

    @staticmethod
    def load_context_examples(context_file, max_context_examples=10, seed=42) -> Optional[List[str]]:
        """
        Loads the context sentences from the given file.
        """
        
        if context_file == 'dummy':       
            context_examples = [
                'Am I a teacher?',
                'Are you from France?',
                'Is she tall?',
                'Do you like pizza?',
                'Does he have a brother?',
                'Did you eat breakfast this morning?',
                'Is it going to rain tonight?',
                'Were you on holidays last week?',
                'Will you attend university next year?',
                'Was he nice?',
                ]
            logger.info(f'using {len(context_examples)} dummy context examples')
        
        elif Path(context_file).is_file():

            context_examples = []
            with open(context_file, 'r', encoding='utf8') as f:
                for line in tqdm(f):
                    context_examples.append(line.strip())
            max_context_examples = min(max_context_examples, len(context_examples))
            random.seed(seed)
            context_examples = list(random.sample(context_examples, max_context_examples))
            logger.info(f'loaded {len(context_examples)} context examples')

        else:
            context_examples = None

        return context_examples

if __name__ == "__main__":

    parser = HfArgumentParser((InferenceArguments, ModelArguments, DataTrainingArguments))
    m = InferenceModel(parser)
    predict_dataset = m.load_test_set_for_generation() # default: data/Topical-Chat/KGD/test_freq.json
    # predict_dataset2 = m.load_test_set_for_generation('data/Topical-Chat/KGD/test_rare.json')

    # cross attention generation
    outputs = m.generate_KGD(predict_dataset)

    # m.write_outputs_to_outfile(outputs, m.gen_args.outfile)