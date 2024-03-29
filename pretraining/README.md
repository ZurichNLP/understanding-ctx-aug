# Pretraining Mini BART Models

BART pre-training was recently implemented in Hugging Face, but uses FLAX and doesn't implement **all** of the functionality of the original denoising dataset.

This repo (https://github.com/kb-labb/kb_bart) has some useful information on preparing data for training BART models with Fairseq using Hugging Face tokenizers. 

We opt for the original Fairseq implementation for pre-training and use a Hugging Face tokenizer for preprocessing. Once pre-trained, we convert the model to Hugging Face for fine-tuning.

## Steps to Reproduce

For simplicity, first create a symlink to the resources folder containing models and data directories.

### 1. Download Data

We use the version of [Bookcorpus collected by Shawn Presser](https://twitter.com/theshawwn/status/1301852133319294976). To download, run:

```bash
bash get_data.sh
```

### 2. Prepare For Fairseq

As preprocessing, we perform the following:
    - extract 'decent' looking sentences from each book and write them to train/validation splits
    - train a tokenizer on the training set
    - apply tokenizer to training and validation sets
    - binarize d/datata ready for fairseq

```bash
sbatch jobs/run_data_prep.sh -r /net/cephfs/data/tkew/projects/unsup_cntrl -d resources/data/books1 # ~ 12 hours
# note, paths to sub-scripts may need to be adjusted in `prepare_bookcorpus.sh`
```

### 3. Run Pretraining

Using the custom `bart-small` config, you can pre-train with different configs defined in `jobs/run_pretraining.sh`. Note, We also include conversion to Hugging Face after pre-training.

The main parameters for denoising objectives are:
- `replace-length` : 0 = no mask, 1 = 1 x `<mask>` for m tokens, -1 `<mask>` for each token, baseline=1
- `rotate`: document rotation: not used in final BART models, baseline=0.0
- `mask-random`: instead of using `<mask>`, use random token this often, baseline=0.1
- `permute-sentences`: entence permutation: portion of sentences that are randomly shuffled in batch, baseline=1.0
- `insert`: insert this percentage of additional random tokens, baseline=0.0
- `poisson-lambda`: defined in paper as lambda=3, baseline=3.0
- `mask`: portion of words/subwords that will be masked, baseline=0.3

```bash
sbatch jobs/run_pretraining.sh -r /net/cephfs/data/tkew/projects/unsup_cntrl -p sm_baseline
# configs: sm_baseline, sm_no_permute, sm_no_masking, sm_w_rotate
```
<!-- ```bash
. ./train_bart_fairseq.sh && train_baseline # ~ 6 hours
``` -->

### 4. Convert to Hugging Face

```bash
python convert_fairseq_model_to_transformers.py \
    --checkpoint resources/models/pt/fairseq/bart_small/Rl1Mr01Rt0Ps1In0Pl3Ma03 \
    --tokenizer resources/data/books1/tok/tokenizer \
    --out_dir resources/models/pt/hf_conv/bart_small/Rl1Mr01Rt0Ps1In0Pl3Ma03
```

We also provide a script to run the conversion on a slurm cluster, e.g.

```bash
sbatch jobs/run_conversion.sh \
    -r /net/cephfs/data/tkew/projects/unsup_cntrl
    -c resources/models/pt/fairseq/bart_small/Rl1Mr01Rt0Ps1In0Pl3Ma03 \
    -t resources/data/books1/tok/tokenizer \
    -o resources/models/pt/hf_conv/bart_small/Rl1Mr01Rt0Ps1In0Pl3Ma03
```
