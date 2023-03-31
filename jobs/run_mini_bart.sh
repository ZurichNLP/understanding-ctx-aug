#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# example usage:
# . jobs/run_mini_bart.sh -s 85 -c exp_configs/SI_t5.yml -s 1

BASE='/data/tkew/projects/unsup_ctrl/'
FORCE=0 # whether to overwrite existing files
SEED=4 # default

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -c config yml [-b repo base] [-s seed] [-f force]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "b:s:c:f:" flag; do
  case "${flag}" in
    s) SEED="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    c) CONFIG_YML="$OPTARG" ;;
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

if [[ -z $CONFIG_YML ]]; then
    print_missing_arg "[-c CONFIG_YML]" "config yml"
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

#######################################################################
# GET EXPERIMENT SETTINGS 
#######################################################################

eval "$(parse_yaml "$CONFIG_YML")"

PT_CONFIG=$(basename "$CONFIG_YML")
PT_CONFIG="${PT_CONFIG%.*}"

# #######################################################################
# # SLURM JOB ARGS
# #######################################################################

SLURM_ARGS_GENERIC="--cpus-per-task=1 --time=00:10:00 --mem=4G --partition=generic"
SLURM_ARGS_VOLTA="--qos=vesta --time=10:00:00 --gres gpu:1 --cpus-per-task 1 --mem-per-cpu=8G --partition=volta"
# SLURM_ARGS_DUMMY="--qos=vesta --time=0:10:00 --gres gpu:1 --cpus-per-task 1 --mem 8g"


# #######################################################################
# # SET EXPERIMENT SETTINGS 
# #######################################################################

DENOISING_ARGS="--replace-length=$REPLACE_LENGTH --mask-random=$MASK_RANDOM --rotate=$ROTATE --permute-sentences=$PERMUTE_SENTENCES --insert=$INSERT --poisson-lambda=$POISSON_LAMBDA --mask=$MASK"
echo "Denoising args: $DENOISING_ARGS"

# get save_dir name
DENOISING_ID=$(parse_denoising_args_to_string "$DENOISING_ARGS")
echo "Denoising args: $DENOISING_ID"

MODEL_ID="$MODEL_CONFIG-$PT_CONFIG" # simplified model id based on config yml
LOG_DIR="$SAVE_DIR_PREFIX/seed_$SEED/logs/$MODEL_ID"
PRETRAIN_SAVE_DIR="$SAVE_DIR_PREFIX/seed_$SEED/pt/fairseq/$MODEL_ID"
CONVERT_SAVE_DIR="$SAVE_DIR_PREFIX/seed_$SEED/pt/hf_conv/$MODEL_ID"
FINETUNE_SAVE_DIR="$SAVE_DIR_PREFIX/seed_$SEED/ft/$MODEL_ID"
RESULTS_DIR="$SAVE_DIR_PREFIX/seed_$SEED/results"

#######################################################################
# INIT LOGGING
#######################################################################

SLURM_DEFAULT_FILE_PATTERN="%j.out"
SLURM_LOG_ARGS="-o $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN -e $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN"

if [[ ! -d "$PRETRAIN_SAVE_DIR" ]]; then
    mkdir -p "$PRETRAIN_SAVE_DIR" "$CONVERT_SAVE_DIR" "$FINETUNE_SAVE_DIR" "$LOG_DIR" "$RESULTS_DIR"
elif [[ "$FORCE" == 1 ]]; then
    echo "Overwriting existing directory $PRETRAIN_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
    rm -rf "$PRETRAIN_SAVE_DIR" && mkdir -p "$PRETRAIN_SAVE_DIR"
    rm -rf "$CONVERT_SAVE_DIR" && mkdir -p "$CONVERT_SAVE_DIR"
    rm -rf "$FINETUNE_SAVE_DIR" && mkdir -p "$FINETUNE_SAVE_DIR"
    # rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR" # don't delete old logs
else
    echo "PRETRAIN_SAVE_DIR already exists: $PRETRAIN_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
    exit 1
fi

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

# log key info
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
date | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"
echo "BASE: $BASE" | tee -a "$LOG_DIR/MAIN"
echo "SEED: $SEED" | tee -a "$LOG_DIR/MAIN"
echo "MODEL_CONFIG: $MODEL_CONFIG" | tee -a "$LOG_DIR/MAIN"
echo "TASK: $TASK" | tee -a "$LOG_DIR/MAIN"
echo "MODEL_ID: $MODEL_ID" | tee -a "$LOG_DIR/MAIN"
echo "DENOISING_METHOD: $DENOISING_METHOD" | tee -a "$LOG_DIR/MAIN"
echo "HEADS_PROB: $HEADS_PROB" | tee -a "$LOG_DIR/MAIN"
echo "DENOISING_ARGS: $DENOISING_ARGS" | tee -a "$LOG_DIR/MAIN"
echo "DENOISING_ID: $DENOISING_ID" | tee -a "$LOG_DIR/MAIN"
echo "PRETRAIN_SAVE_DIR: $PRETRAIN_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
echo "CONVERT_SAVE_DIR: $CONVERT_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
echo "FINETUNE_SAVE_DIR: $FINETUNE_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
echo "RESULTS_DIR: $RESULTS_DIR" | tee -a "$LOG_DIR/MAIN"
echo "LOG_DIR: $LOG_DIR" | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"

# run pretraining
id_pretrain=$(
    $BASE/jobs/sbatch_bare.sh \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_pretraining.sh \
    --data_dir "$DATA_DIR" \
    --save_dir "$PRETRAIN_SAVE_DIR" \
    --model_config "$MODEL_CONFIG" \
    --task "$TASK" \
    --seed "$SEED" \
    --replace_length "$REPLACE_LENGTH" \
    --mask_random "$MASK_RANDOM" \
    --rotate "$ROTATE" \
    --permute_sentences "$PERMUTE_SENTENCES" \
    --insert "$INSERT" \
    --poisson_lambda "$POISSON_LAMBDA" \
    --mask "$MASK" \
    --denoising_method "$DENOISING_METHOD" \
    --heads_prob "$HEADS_PROB"
    )

echo "  id_pretrain: $id_pretrain | $LOG_DIR/$id_pretrain.out" | tee -a "$LOG_DIR/MAIN"

# run fs -> hf conversion
id_convert=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_pretrain \
    $SLURM_ARGS_GENERIC \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_model_conversion.sh \
    -c "$PRETRAIN_SAVE_DIR" -t "$DATA_DIR/../tok/tokenizer" -o "$CONVERT_SAVE_DIR"       
)

echo "  id_convert: $id_convert | $LOG_DIR/$id_convert.out" | tee -a "$LOG_DIR/MAIN"

id_finetune=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_convert \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$CONVERT_SAVE_DIR" -o "$FINETUNE_SAVE_DIR" -s "$SEED"
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