#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Job script to fine-tune a public model (e.g. BART-base) on the Topical-Chat dataset and evaluate it on the test set.

# example usage:
# . jobs/run_public.sh -s 23 -m t5-small -f 1

# updates:
# - 04.04.23: multile dataset experiments

BASE='/data/tkew/projects/unsup_ctrl/'
FORCE=0 # whether to overwrite existing files
SEED=4 # default
SAVE_DIR_PREFIX="$BASE/resources/models"
DATA_DIR="$SAVE_DIR_PREFIX/../data/Topical-Chat/KGD/"

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script [-b base] [-s seed] [-m model] [-f force] -d [data_dir]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "b:s:m:f:d:" flag; do
  case "${flag}" in
    s) SEED="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    m) HF_MODEL_NAME="$OPTARG" ;;
    f) FORCE="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $BASE ]]; then
    print_missing_arg "[-b BASE]" "Base directory of the repository"
    exit 1
fi

if [[ -z $HF_MODEL_NAME ]]; then
    print_missing_arg "[-m HF_MODEL_NAME]" "Model ID from Hugging Face"
    exit 1
fi

if [[ -z $SEED ]]; then
    print_missing_arg "[-s SEED]" "random seed"
    exit 1
fi

# cd to base dir
cd "$BASE" && echo $(pwd) || exit 1

#######################################################################
# ACTIVATE ENV
#######################################################################

source "$BASE/start.sh"

#######################################################################
# IMPORT HELPERS
#######################################################################

source "$BASE/jobs/job_utils.sh" 

# #######################################################################
# # SLURM JOB ARGS
# #######################################################################

# SLURM_ARGS_VOLTA="--time=12:00:00 --gres=gpu:1 --cpus-per-task=1 --mem-per-cpu=8G --partition=lowprio"
# SLURM_ARGS_VOLTA_LARGE="--time=12:00:00 --gres=gpu:V100:1 --constraint=GPUMEM32GB --cpus-per-task=1 --mem-per-cpu=16G --partition=lowprio"

# #######################################################################
# # SET EXPERIMENT SETTINGS 
# #######################################################################

case $HF_MODEL_NAME in
    "facebook/bart-base")
        MODEL_ID="bart_base"
        ;;
    "facebook/bart-large")
        MODEL_ID="bart_large"
        ;;
    "t5-small")
        MODEL_ID="t5_small"
        ;;
    "t5-base")
        MODEL_ID="t5_base"
        ;;
    "google/t5-v1_1-small")
        MODEL_ID="t5v11_small"
        ;;
    "google/t5-v1_1-base")
        MODEL_ID="t5v11_base"
        ;;
    "google/t5-small-lm-adapt")
        MODEL_ID="t5_lm_small"
        ;;
    "google/t5-base-lm-adapt")
        MODEL_ID="t5_lm_base"
        ;;
    "resources/models/mass/hf_conv")
        MODEL_ID="mass_base"
        ;;
    "resources/models/fairseq_mass/hf_conv")
        MODEL_ID="mass_base"
        ;;
    *)
        echo -n "unknown model name: $HF_MODEL_NAME" && exit 1
    ;;
esac

case $DATA_DIR in
    "resources/data/DailyDialog/DD")
        TEST_SET="$DATA_DIR/test.json"
        ;;
    "resources/data/Commonsense-Dialogues/CD")
        TEST_SET="$DATA_DIR/test.json"
        ;;
    "resources/data/Topical-Chat/KGD")
        TEST_SET="$DATA_DIR/test_freq.json" # can also use test_rare.json
        ;;
    *)
        echo -n "failed to identify test set in $DATA_DIR" && exit 1
    ;;
esac

DATASET_ID=$(infer_dataset_id $DATA_DIR)
FINETUNED_MODEL_DIR=$(infer_model_path $SEED $DATASET_ID $MODEL_ID)
RESULTS_DIR=$(infer_output_path $FINETUNED_MODEL_DIR $TEST_SET)
LOG_DIR="$FINETUNED_MODEL_DIR/logs"

echo "$FINETUNED_MODEL_DIR"
echo "$LOG_DIR"

######################################################################
INIT LOGGING
######################################################################

SLURM_DEFAULT_FILE_PATTERN="%j.out"
SLURM_LOG_ARGS="-o $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN -e $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN"

mkdir -p "$FINETUNED_MODEL_DIR" "$LOG_DIR"

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

# log key info
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
date | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
echo "SLURM_ARGS: $SLURM_ARGS_VOLTA_LARGE $SLURM_LOG_ARGS" | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
echo "BASE: $BASE" | tee -a "$LOG_DIR/MAIN"
echo "SEED: $SEED" | tee -a "$LOG_DIR/MAIN"
echo "HF_MODEL_NAME: $HF_MODEL_NAME" | tee -a "$LOG_DIR/MAIN"
echo "MODEL_ID: $MODEL_ID" | tee -a "$LOG_DIR/MAIN"
echo "DATA_DIR: $DATA_DIR" | tee -a "$LOG_DIR/MAIN"
echo "FINETUNED_MODEL_DIR: $FINETUNED_MODEL_DIR" | tee -a "$LOG_DIR/MAIN"
echo "RESULTS_DIR: $RESULTS_DIR" | tee -a "$LOG_DIR/MAIN"
echo "TEST_SET: $TEST_SET" | tee -a "$LOG_DIR/MAIN"
echo "LOG_DIR: $LOG_DIR" | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"

# run fine-tuning
id_finetune=$(
    $BASE/jobs/sbatch_bare.sh \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$HF_MODEL_NAME" -o "$FINETUNED_MODEL_DIR" -s "$SEED" -d "$DATA_DIR"
)

echo "  id_finetune: $id_finetune | $LOG_DIR/$id_finetune.out" | tee -a "$LOG_DIR/MAIN"

# run generation/evaluation
id_generate=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_finetune \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_generation_exp.sh \
    -m "$FINETUNED_MODEL_DIR" -b 120 -t "$TEST_SET" -o "$RESULTS_DIR"
)

echo "  id_generate: $id_generate | $LOG_DIR/$id_generate.out" | tee -a "$LOG_DIR/MAIN"