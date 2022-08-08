
## 

This repository is a reimplementation of zero-shot control methods described in the following papers by Hazarika et al.:

- Attention Biasing and Context Augmentation for Zero-Shot Control of Encoder-Decoder Transformers for Natural Language Generation (2022)
- Zero-Shot Controlled Generation with Encoder-Decoder Transformers (2021/2022)



## Setup

```
conda create -n unsup_ctrl python=3.8
conda activate unsup_ctrl
pip install -r requirements.txt

# for preprocessing
python -m spacy download en_core_web_sm

# cd src/transformers
# git checkout origin/unsup_cntrl
```

### Data

Experiments in the original paper mostly use the Topical Chat dataset ([https://m.media-amazon.com/images/G/01/amazon.jobs/3079_Paper._CB1565131710_.pdf](Gopalakrishnan_et_al_2019)). This dataset can be found at [https://github.com/alexa/Topical-Chat]

To prepare the data, do the following:

```
mkdir data
cd data
git clone https://github.com/alexa/Topical-Chat.git
cd Topical-Chat/src
pip3 install -r requirements.txt

# NOTE: Building the data requires Reddit credentials. 
# Please create your own Reddit API keys: https://www.reddit.com

# NOTE: To collect the reading sets, the IDs pointing to one data point has changed (https://github.com/alexa/Topical-Chat/issues/11),
# so you need to change the ID "t3_2au72q" to "t3_r8dxya" in the following files:
# reading_sets/pre-build/test_freq.json, reading_sets/pre-build/train.json, reading_sets/pre-build/valid_freq.json

python3 build.py  --reddit_client_id CLIENT_ID --reddit_client_secret CLIENT_SECRET --reddit_user_agent USER_AGENT
```

This build takes around 1 hour. Once completed, we can prepare the data for training according to the desctiption provided in Hazarika et al., (2021)with the following command:

```
bash prepare_data.sh
```

<!-- ```
python prepare_topical_chat_dataset.py --data_dir data/Topical-Chat --split test_freq
``` -->

### Fine-tuning base models

The script `train.py` is adapted from Hugging Face's `run_summarization.py` example script and can be used to fine-tune a new model for our experiments.

The script `run_finetuning.sh` provides the training commands used to train our models. For example, to re-run fine-tuning for BART-base, run

```
. ./run_finetuning.sh && finetune_bart_base_for_kgd [GPU_ID]
```

### Zero-shot Controlled Generation

To run inference, run:

```
# baseline (no zero-shot control knobs)
python inference.py \
    --model_name_or_path "models/bart-base" \
    --checkpoint_dir "checkpoint-21786" \
    --test_file data/Topical-Chat/KGD/test_freq.json \
    --text_column "turns" --summary_column "target" --knowledge_column "knowledge" \
    --seed 42 --batch_size 120 \
    --num_return_sequences 1 --beam_size 4 \
    --do_sample True --top_p 0.9 \
    --write_to_file auto

# experimental
python inference.py \
    --model_name_or_path "models/bart-base" \
    --checkpoint_dir "checkpoint-21786" \
    --test_file data/Topical-Chat/KGD/test_freq.json \
    --text_column "turns" --summary_column "target" --knowledge_column "knowledge" \
    --seed 42 --batch_size 120 \
    --num_return_sequences 1 --beam_size 4 \
    --do_sample True --top_p 0.9 \
    --cross_attention_bias 5 --cross_attention_mode knowledge \
    --context_augmentation_examples data/Topical-Chat/KGD/contexts/questions.txt --context_code_attention_bias 10  --max_context_examples 10 \
    --write_to_file auto



note, set:
    --max_predict_samples # if debugging or just running on a subset of examples
```

### Evaluate


```
# baseline
python evaluate_zero_shot.py models/bart-base/checkpoint-21786/generations_seed=42_ml=64_lp=1.0_ns=1_bs=4_ds=1_temp=1.0_tk=0_tp=0.9_xatt=1-uniform-_ctxt=1--10.txt

# questions context conditioned
python evaluate_zero_shot.py models/bart-base/checkpoint-21786/generations_seed=42_ml=64_lp=1.0_ns=1_bs=4_ds=1_temp=1.0_tk=0_tp=0.9_xatt=5-uniform-knowledge_ctxt=10-questions-10.txt
```

<!-- **TODO**

```
# with MUSS simplification model (ported to HF):
python test_run.py /scratch/tkew/ctrl_tokens/resources/models/muss_en_mined_hf

``` -->

