#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Job script to train a public model (e.g. BART-base) from scratch on the Topical-Chat dataset and evaluate it on the test set.

# example usage:
# . jobs/run_public.sh -s 23 -m t5-small -f 1

BASE='/net/cephfs/data/tkew/projects/unsup_cntrl'
FORCE=0 # whether to overwrite existing files
SEED=4 # default
SAVE_DIR_PREFIX="$BASE/resources/models"

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script "
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "b:s:m:f:" flag; do
  case "${flag}" in
    s) SEED="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    m) HF_MODEL_NAME="$OPTARG" ;;
    f) FORCE="$OPTARG" ;;
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

SLURM_ARGS_VOLTA="--qos=vesta --time=12:00:00 --gres gpu:1 --cpus-per-task 1 --mem-per-cpu=8G --partition=volta"
SLURM_ARGS_VOLTA_LARGE="--qos=vesta --time=12:00:00 --gres gpu:Tesla-V100-32GB:1 --cpus-per-task 1 --mem-per-cpu=16G --partition=volta"

# #######################################################################
# # SET EXPERIMENT SETTINGS 
# #######################################################################

case $HF_MODEL_NAME in
    "t5-small")
        MODEL_ID="t5_small"
        ;;
    "google/t5-small-lm-adapt")
        MODEL_ID="t5_lm_small"
        ;;
    "facebook/bart-base")
        MODEL_ID="bart_base"
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


LOG_DIR="$SAVE_DIR_PREFIX/seed_$SEED/logs/$MODEL_ID"
FINETUNE_SAVE_DIR="$SAVE_DIR_PREFIX/seed_$SEED/ft/$MODEL_ID"
RESULTS_DIR="$SAVE_DIR_PREFIX/seed_$SEED/results_topchat_kgd_test_freq"

echo "$LOG_DIR"
echo "$FINETUNE_SAVE_DIR"
echo "$RESULTS_DIR"


#######################################################################
# INIT LOGGING
#######################################################################

SLURM_DEFAULT_FILE_PATTERN="%j.out"
SLURM_LOG_ARGS="-o $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN -e $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

# log key info
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
date | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
echo "BASE: $BASE" | tee -a "$LOG_DIR/MAIN"
echo "SEED: $SEED" | tee -a "$LOG_DIR/MAIN"
echo "HF_MODEL_NAME: $HF_MODEL_NAME" | tee -a "$LOG_DIR/MAIN"
echo "MODEL_ID: $MODEL_ID" | tee -a "$LOG_DIR/MAIN"
echo "FINETUNE_SAVE_DIR: $FINETUNE_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
echo "RESULTS_DIR: $RESULTS_DIR" | tee -a "$LOG_DIR/MAIN"
echo "LOG_DIR: $LOG_DIR" | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"

id_finetune=$(
    $BASE/jobs/sbatch_bare.sh \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$HF_MODEL_NAME" -o "$FINETUNE_SAVE_DIR" -s "$SEED"
)

echo "  id_finetune: $id_finetune | $LOG_DIR/$id_finetune.out" | tee -a "$LOG_DIR/MAIN"

# run generation/evaluation
id_generate=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_finetune \
    $SLURM_ARGS_VOLTA_LARGE \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_generation_exp.sh \
    -m "$FINETUNE_SAVE_DIR" -b 120 -o "$RESULTS_DIR"
)

echo "  id_generate: $id_generate | $LOG_DIR/$id_generate.out" | tee -a "$LOG_DIR/MAIN"