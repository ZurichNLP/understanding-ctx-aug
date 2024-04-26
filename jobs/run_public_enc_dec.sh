#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# example usage:
# . jobs/dummy_run.sh -s 85 -c configs/exp1.yml -f 1

BASE='/data/tkew/projects/understanding-ctx-aug/'
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

SLURM_ARGS_VOLTA="--time=12:00:00 --gres=gpu:1 --cpus-per-task=1 --mem-per-cpu=8G --partition=lowprio"
SLURM_ARGS_VOLTA_LARGE="--time=12:00:00 --gres=gpu:V100:1 --constraint=GPUMEM32GB --cpus-per-task=1 --mem-per-cpu=16G --partition=lowprio"

# #######################################################################
# # SET EXPERIMENT SETTINGS 
# #######################################################################

case $HF_MODEL_NAME in
    "roberta-base")
        MODEL_ID="roberta_base"
        ;;
    "bert-base-cased")
        MODEL_ID="bert_base"
        ;;
    *)
        echo -n "unknown model name: $HF_MODEL_NAME" && exit 1
    ;;
esac


LOG_DIR="$SAVE_DIR_PREFIX/seed_$SEED/logs/$MODEL_ID"
FINETUNE_SAVE_DIR_1="$SAVE_DIR_PREFIX/seed_$SEED/ft/${MODEL_ID}_shared"
FINETUNE_SAVE_DIR_2="$SAVE_DIR_PREFIX/seed_$SEED/ft/${MODEL_ID}"
RESULTS_DIR="$SAVE_DIR_PREFIX/seed_$SEED/results"

echo "$LOG_DIR"
echo "$FINETUNE_SAVE_DIR_1"
echo "$FINETUNE_SAVE_DIR_2"
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
echo "FINETUNE_SAVE_DIR_1: $FINETUNE_SAVE_DIR_1" | tee -a "$LOG_DIR/MAIN"
echo "FINETUNE_SAVE_DIR_2: $FINETUNE_SAVE_DIR_2" | tee -a "$LOG_DIR/MAIN"
echo "RESULTS_DIR: $RESULTS_DIR" | tee -a "$LOG_DIR/MAIN"
echo "LOG_DIR: $LOG_DIR" | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"

id_finetune_1=$(
    $BASE/jobs/sbatch_bare.sh \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$HF_MODEL_NAME" -o "$FINETUNE_SAVE_DIR_1" -s "$SEED" --is_encoder_decoder True --tie_encoder_decoder True
)

echo "  id_finetune_1: $id_finetune_1 | $LOG_DIR/$id_finetune_1.out" | tee -a "$LOG_DIR/MAIN"

# run generation/evaluation
id_generate_1=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_finetune_1 \
    $SLURM_ARGS_VOLTA_LARGE \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_generation_exp.sh \
    -m "$FINETUNE_SAVE_DIR_1" -b 120 -o "$RESULTS_DIR"
)

echo "  id_generate_1: $id_generate_1 | $LOG_DIR/$id_generate_1.out" | tee -a "$LOG_DIR/MAIN"

id_finetune_2=$(
    $BASE/jobs/sbatch_bare.sh \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$HF_MODEL_NAME" -o "$FINETUNE_SAVE_DIR_2" -s "$SEED" --is_encoder_decoder True --tie_encoder_decoder False
)

echo "  id_finetune_2: $id_finetune_2 | $LOG_DIR/$id_finetune_2.out" | tee -a "$LOG_DIR/MAIN"

id_generate_2=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_finetune_2 \
    $SLURM_ARGS_VOLTA_LARGE \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_generation_exp.sh \
    -m "$FINETUNE_SAVE_DIR_2" -b 120 -o "$RESULTS_DIR"
)

echo "  id_generate_2: $id_generate_2 | $LOG_DIR/$id_generate_2.out" | tee -a "$LOG_DIR/MAIN"
