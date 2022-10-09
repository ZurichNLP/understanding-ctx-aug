#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collection of dataclasses for hyperparameter arguments
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )

    is_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "Set this flag if you are training an encoder-decoder model."},
    )

    tie_encoder_decoder: Optional[bool] = field(
        default=False,
        metadata={"help": "to create a shared encoder-decoder model, set this to True"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )

    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )

    knowledge_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the knowledge snippet (for knowledge-grounded dialogue gen)."},
    )

    knowledge_bucket_size: Optional[int] = field(
        default=32,
        metadata={"help": "The number of tokens in the knowledge bucket (for knowledge-grounded dialogue gen)."},
    )

    history_bucket_size: Optional[int] = field(
        default=25,
        metadata={"help": "The number of tokens in the history bucket (for knowledge-grounded dialogue gen)."},
    )

    speaker_id_tag: Optional[str] = field(
        default='<speaker1>',
        metadata={"help": "The tag used to indicate the speaker in the knowledge column (for knowledge-grounded dialogue gen)."},
    )

    project_name: Optional[str] = field(
        default='unsup_ctrl',
        metadata={"help": "Project name used for logging to WandB."},
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )

    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )

    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    persist_datasets: bool = field(
        default=False, metadata={"help": "Save the preprocessed datasets to disk as json lines. This will take more space."}
    )

    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )

    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    max_eval_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_predict_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    early_stopping: bool = field(
        default=False,
        metadata={
            "help": "whether or not to monitor for early stopping"
        },
    )

    early_stopping_patience: Optional[int] = field(
        default=3,
        metadata={
            "help": "number of eval steps to run before terminating due to no improvement"
        },
    )
    
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "how much the specified metric must improve to satisfy early stopping conditions"
        },
    )

    eval_runs_per_epoch: Optional[int] = field(
        default=1,
        metadata={
            "help": "number of eval runs to perform per epoch (for experimental purposes). Note, this will override the `eval_steps` and `save_steps` argument."
        },
    )

    write_intermediate_eval_results: bool = field(
        default=False,
        metadata={
            "help": "whether or not to write intermediate eval results to disk. If True, will save results to json files in the output_dir"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

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

    # use_cross_attention_bias: bool = field(
    #     default=False,
    #     metadata={"help": "Use cross attention"}
    # )

    cross_attention_bias_value: int = field(
        default=1,
        metadata={"help": "Value used to bias cross attention. Default = 1, i.e. no bias. "
            "A bias of 0 acts similarly to setting a custom attention mask for the cross attention."}
    )

    context_augmentation_examples: str = field(
        default=None,
        metadata={"help": "source for context examples if using context augmentation as described by Hazarika et al., 2021. "
        "If a file path is provided, example sentences are expected to be one per line"}
    )

    max_context_examples: int = field(
        default=10,
        metadata={"help": "number of context examples to use for context augmentation"}
    )

    context_code_attention_bias_value: int = field(
        default=1,
        metadata={"help": "Value used to bias cross attention given context augmentation. Default = 1, i.e. no bias. "
            "A bias of 0 acts similarly to setting a custom attention mask for the cross attention."}
    )

    bias_profile: str = field(
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

    verbose: bool = field(
        default=False,
        metadata={"help": "Print progress"}
    )

    debug: bool = field(
        default=False,
        metadata={"help": "Print debug information"}
    )


if __name__ == "__main__":
    pass
