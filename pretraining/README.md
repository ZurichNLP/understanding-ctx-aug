## Pretraining BART

BART pre-training was recently implemented in Hugging Face, but uses FLAX and doesn't implement **all** of the functionality of the original denoising dataset.

This repo (https://github.com/kb-labb/kb_bart) has some useful information on preparing data for training BART models with Fairseq using Hugging Face tokenizers. 

We opt for the original Fairseq implementation for pre-training and use a Hugging Face tokenizer for preprocessing. Once pre-trained, we convert the model to Hugging Face for fine-tuning.

```
srun -p volta --pty -n 1 -c 4 --time=10:00:00 --gres gpu:1 --mem=8G bash -l

tmux new -s nested

. start.sh

python run_dlm.py     --model_name_or_path "scratch/cust_bart_small"     --train_file scratch/bookcorpus/tmp/train.txt     --validation_file scratch/bookcorpus/tmp/valid.txt     --per_device_train_batch_size 8     --per_device_eval_batch_size 8     --do_train     --do_eval     --overwrite_output_dir --output_dir "scratch/cust_bart_small_pt" --max_seq_len 512 --report_to none

```


## Steps to reproduce

#### 1. Prepare Data For Fairseq

We use the version of Bookcorpus collected by Shawn Presser (https://twitter.com/theshawwn/status/1301852133319294976)

To download, run `wget https://battle.shawwn.com/sdb/books1/books1.tar.gz`.

Preprocessing involves the following steps:
    - extracts decent looking sentences from each book and write them to train/validation splits
    - trains a tokenizer on the training set
    - applies tokenizer to training and validation set
    - binarizes data ready for fairseq

```bash
sbatch jobs/run_data_prep.sh -d resources/data/books1 # ~ 12 hours
```

#### 2. 

```bash
. ./train_bart_fairseq.sh && train_baseline # ~ 6 hours
```

#### 3. Convert Fairseq model to Hugging Face

```bash
python convert_fairseq_model_to_transformers.py \
    --checkpoint scratch/models/fairseq/bart_small/checkpoint_best.pt \
    --tokenizer scratch/books1/tok/tokenizer \
    --out_dir scratch/models/huggingface_conv/bart_small
```

#### 4. Finetune pretrained model on KGD task

