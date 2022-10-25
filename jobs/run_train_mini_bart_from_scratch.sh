#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Job script to train a Bart-mini model from scratch on the Topical-Chat dataset and evaluate it on the test set.

# example usage:
# bash jobs/run_train_mini_bart_from_scratch.sh -s 23 -c resources/models/seed_23/pt/hf_conv/bart_small-PS -m bart_small_rndm

BASE='/net/cephfs/data/tkew/projects/unsup_cntrl'
SEED=4 # default
FORCE=0 # whether to overwrite existing files

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -c HF model config [-b repo base] [-s seed] [-f force]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "b:s:c:m:o:f:" flag; do
  case "${flag}" in
    s) SEED="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    c) CONFIG_PATH="$OPTARG" ;;
    m) MODEL_ID="$OPTARG" ;;
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

if [[ -z $CONFIG_PATH ]]; then
    print_missing_arg "[-c CONFIG_PATH]" "Pre-trained model name or path"
    exit 1
fi

if [[ -z $MODEL_ID ]]; then
    print_missing_arg "[-m MODEL_ID]" "local model name identifier"
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

# #######################################################################
# # SLURM JOB ARGS
# #######################################################################

SLURM_ARGS_VOLTA="--qos=vesta --time=10:00:00 --gres gpu:1 --cpus-per-task 1 --mem-per-cpu=8G --partition=volta"

# #######################################################################
# # SET EXPERIMENT SETTINGS 
# #######################################################################

SAVE_DIR_PREFIX="resources/models"
LOG_DIR="$SAVE_DIR_PREFIX/seed_$SEED/logs/$MODEL_ID"
FINETUNE_SAVE_DIR="$SAVE_DIR_PREFIX/seed_$SEED/ft/$MODEL_ID"
RESULTS_DIR="$SAVE_DIR_PREFIX/seed_$SEED/results_topchat_kgd_test_freq"


if [[ ! -d "$FINETUNE_SAVE_DIR" ]]; then
    mkdir -p "$FINETUNE_SAVE_DIR" "$LOG_DIR" "$RESULTS_DIR"
elif [[ "$FORCE" == 1 ]]; then
    echo "Overwriting existing directory $FINETUNE_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
    rm -rf "$FINETUNE_SAVE_DIR" && mkdir -p "$FINETUNE_SAVE_DIR"
    # rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR" # don't delete old logs
else
    echo "FINETUNE_SAVE_DIR already exists: $FINETUNE_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
    exit 1
fi

#######################################################################
# INIT LOGGING
#######################################################################

SLURM_DEFAULT_FILE_PATTERN="%j.out"
SLURM_LOG_ARGS="-o $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN -e $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN"

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

# log key info
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
date | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
echo "BASE: $BASE" | tee -a "$LOG_DIR/MAIN"
echo "SEED: $SEED" | tee -a "$LOG_DIR/MAIN"
echo "MODEL_ID: $MODEL_ID" | tee -a "$LOG_DIR/MAIN"
echo "CONFIG_PATH: $CONFIG_PATH" | tee -a "$LOG_DIR/MAIN"
echo "FINETUNE_SAVE_DIR: $FINETUNE_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
echo "RESULTS_DIR: $RESULTS_DIR" | tee -a "$LOG_DIR/MAIN"
echo "LOG_DIR: $LOG_DIR" | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"

id_finetune=$(
    $BASE/jobs/sbatch_bare.sh \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$CONFIG_PATH" -o "$FINETUNE_SAVE_DIR" -s "$SEED" --init_as_random True
)

echo "  id_finetune: $id_finetune | $LOG_DIR/$id_finetune.out" | tee -a "$LOG_DIR/MAIN"

# run generation/evaluation
id_generate=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_finetune \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_generation_exp_parallel.sh \
    -m "$FINETUNE_SAVE_DIR" -b 120 -o "$RESULTS_DIR"
)

echo "  id_generate: $id_generate | $LOG_DIR/$id_generate.out" | tee -a "$LOG_DIR/MAIN"