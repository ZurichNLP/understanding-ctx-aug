# Understanding CTX AUG

This repository contains the code for the ACL Findings paper [Uncovering Hidden Consequences of Pre-training Objectives in Sequence-to-Sequence Models (Kew & Sennrich, 2023)](https://aclanthology.org/2023.findings-acl.438/).

Our experiments reimplement some of the zero-shot control methods described in the papers by [Zero-Shot Controlled Generation with Encoder-Decoder Transformers (Hazarika et al., 2021)](https://arxiv.org/abs/2106.06411) and [Attention Biasing and Context Augmentation for Zero-Shot Control of Encoder-Decoder Transformers for Natural Language Generation (Hazarika et al., 2022)](https://ojs.aaai.org/index.php/AAAI/article/view/21319).

## Setup

We recommend using a clean conda environment to run these scripts.

To set up the working environment, run the following commands.

```bash
# if running on cluster, load the relevant modules, e.g.
module load anaconda3/2022.10 gpu gcc/8.5.0 cudnn/10.2.89

# create new clean environment
conda create -n unsup_ctrl python=3.8 -y
conda activate unsup_ctrl && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

pip install -r requirements.txt

# depending on cuda driver, may need to install from whl, e.g.
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# for finetuning data preprocessing
python -m spacy download en_core_web_sm

# to run notebook from a server with ipython kernels, run
python -m ipykernel install --user --name=unsup_ctrl
```

## Resources

To set up the location of larger files such as data and models:

```bash
mkdir resourses # or ln -s /path/to/storage/ resources
mkdir resources/data
mkdir resources/models
# pretraining resources
ln -s resources pretraining/resources
# We also need a directory for the experiments results:
mkdir results
```

## Data

Experiments in the original paper mostly use the 
**Topical Chat dataset** ([Gopalakrishnan et al., 2019](https://m.media-amazon.com/images/G/01/amazon.jobs/3079_Paper._CB1565131710_.pdf)), 
which can be found [here](https://github.com/alexa/Topical-Chat).

To download the data for fine-tuning, run:

```bash
git clone https://github.com/alexa/Topical-Chat.git data/Topical-Chat
cd data/Topical-Chat/src
pip3 install -r requirements.txt

# NOTE: Building the data requires Reddit credentials. 
# Please create your own Reddit API keys: https://www.reddit.com

# NOTE: To collect the reading sets, the IDs pointing to one data point has changed (https://github.com/alexa/Topical-Chat/issues/11),
# so you need to change the ID "t3_2au72q" to "t3_r8dxya" in the following files:
# reading_sets/pre-build/test_freq.json, reading_sets/pre-build/train.json, reading_sets/pre-build/valid_freq.json

python3 build.py  --reddit_client_id CLIENT_ID --reddit_client_secret CLIENT_SECRET --reddit_user_agent USER_AGENT
```

This build takes around 1 hour. Once completed, we can prepare the data for training according to the description provided in Hazarika et al., (2021) with the following:

```bash
sbatch jobs/run_data_prep-TopicalChat.sh
```

<!-- For additional dataset experiments:

- Commonsense-Dialogues ([Zhou et al., 2021](https://arxiv.org/abs/2109.06427))

```
git clone https://github.com/alexa/Commonsense-Dialogues.git data/Commonsense-Dialogues
sbatch jobs/run_data_prep-CommonsenseDialogue.sh
```

- DailyDialog

```
sbatch jobs/run_data_prep-DailyDialog.sh
``` -->

## Experiments

Experiments were run on a slurm cluster.

To run a controlled experiment with **mini BART models** use [`jobs/run_mini_bart.sh`](./jobs/run_mini_bart.sh), specifying the random seed and the yml config with BART's denoising args.
This performs pre-training, fine-tuning, inference and evaluation.

```bash
bash jobs/run_mini_bart.sh -s 42 -c exp_configs/SI_bart.yml
```

To fine-tune, generate and evaluate a **publicly available pre-trained model** on slurm, use:

```bash
bash jobs/run_public.sh -s 23 -m "facebook/bart-base" -d "resources/data/Topical-Chat/KGD"
bash jobs/run_public.sh -s 23 -m "google/t5-small-lm-adapt" -d "resources/data/Topical-Chat/KGD"
bash jobs/run_public.sh -s 23 -m "t5-small" -d "resources/data/Topical-Chat/KGD"
```

### Individual Steps

#### Pre-training small BART models

See this [README](./pretraining/README.md).

#### Fine-tuning base models

The python script [`./finetune.py`](./finetune.py) is adapted from Hugging Face's `run_summarization.py` example script and can be used to fine-tune a new model for our experiments.

The bash wrapper script [`./finetune.sh`](./finetune.sh) provides the training commands used to train our models.

To fine-tune a model on a slurm cluster use [`jobs/run_finetuning.sh`](./jobs/run_finetuning.sh), e.g.:

```bash
seed=23
sbatch jobs/run_finetuning.sh \
    -i resources/models/seed_$seed/pt/hf_conv/bart_small-MLM/ \
    -o resources/models/seed_$seed/CD/ft/$model_name/ \
    -s $seed \
    -d resources/data/Topical-Chat/KGD
```

#### Inference

To perform inference on a slurm cluster, run:

```bash
sbatch jobs/run_generation_exp.sh \
    -m resources/models/ft/bart_base \
    -t resources/data/Topical-Chat/KGD/test_freq.json
```

For multiple experimental inference runs with BART-mini, it's also possible to parallelise jobs on a single GPU, e.g.

```bash
sbatch jobs/run_generation_exp_parallel.sh \
    -m resources/models/ft/bart_small-MLM \
    -t resources/data/Topical-Chat/KGD/test_freq.json
```

Note: you can modify the experiment IDs in these scripts to match your needs!

#### Inference with ctx. aug. / attn. biasing

The script `constants.py` contains a series of hardcoded experimental configs. 
To run a new experiment (i.e. all seeded generation runs), you can define a new experiment config in this script, e.g.:

```json
"short_qu_ctxt_aug5": {
    "context_augmentation_examples": "resources/data/Topical-Chat/KGD/contexts/short_questions.txt",
    "context_code_attention_bias_value": 5,
    "max_context_examples": 5,
},
```

Note: to avoid errors with post-hoc evaluation (not always used), you should also add the name of the experiment and the relevant filepath ending in `eval.py`.


## Analysing Results

To double check which experiments have been completed and have results, use `check_experiment_results.py`, specifying the dataset ID (TC/CD/DD) and the testset's directory stem, e.g.:

```bash
python check_experiment_results.py TC test_freq-bart_small
```

The results and plots from the paper were generated [`summarize_results.ipynb`](./summarize_results.ipynb) (Note, this notebook hasn't been cleaned!):

<!-- ### Heavy lifting scripts

The commands above assume access to a slurm cluster. For development or direct execution, you can also execute the relevant scripts, specifying the appropriate arguments, e.g.:

```bash
# generation experiments with different seeds
python generation_exp.py -m resources/models/seed_$seed/TC/ft/bart_base --exp_id baseline

# inference (will only generate for the first `max_predict_samples`)
python inference.py \
    --model_name_or_path resources/models/seed_23/CD/ft/bart_small-MLM \
    --test_file resources/data/Commonsense-Dialogues/CD/test.json 
    --text_column turns --summary_column target --knowledge_column context \
    --seed 0 --batch_size 5 --num_return_sequences 1 \
    --beam_size 4 --do_sample True --top_p 0.9 \
    --write_to_file none --max_predict_samples 5 \
    --cross_attention_bias_value 1 --bias_profile knowledge \
    --context_augmentation_examples "resources/data/Commonsense-Dialogues/CD/contexts/train_questions.txt" \
    --context_code_attention_bias_value 5  --max_context_examples 10

# evaluation
python evaluation/evaluation.py output_file [--references_file (e.g., test_freq.json)] [--outfile]
``` -->

## Known Limitations

- The bias profile used is fixed across all decoding timesteps (not gradual)
- Commands for generating all the different types of context example files are missing from this documentation.

## Citation

```
@inproceedings{kew-sennrich-2023-uncovering,
    title = "Uncovering Hidden Consequences of Pre-training Objectives in Sequence-to-Sequence Models",
    author = "Kew, Tannon  and
      Sennrich, Rico",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.438",
    doi = "10.18653/v1/2023.findings-acl.438",
    pages = "7010--7022",
    abstract = "Some variants of self-supervised denoising objectives for pre-training encoder-decoder language models have been reported to have a negligible impact on downstream performance. Yet the design of these pre-training objectives leads to behavioural differences that can be uncovered with specific manipulations. We reproduce a recently proposed zero-shot control method and find that it is only successful on a subset of models. To understand what causes the difference in its effectiveness, we perform a set of controlled experiments, varying only the pre-training objective, and find unexpected interactions between the pre-training method and downstream controllability of models after fine-tuning. Our results show that different pre-training objectives have consequences that may not be visible in standard downstream evaluation, but which should be taken into account when developing models with controllability in mind.",
}
```