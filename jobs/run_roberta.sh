#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# example usage:
# . jobs/dummy_run.sh -s 85 -c configs/exp1.yml -f 1

BASE='/net/cephfs/data/tkew/projects/unsup_cntrl'
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

# OLD_MODEL_ID="$MODEL_CONFIG-${TASK}_$DENOISING_METHOD-$DENOISING_ID"
MODEL_ID="$MODEL_CONFIG-MLM" # simplified model id based on config yml
LOG_DIR="$SAVE_DIR_PREFIX/seed_$SEED/logs/$MODEL_ID"
PRETRAIN_SAVE_DIR="$SAVE_DIR_PREFIX/seed_$SEED/pt/fairseq/$MODEL_ID"
CONVERT_SAVE_DIR="$SAVE_DIR_PREFIX/seed_$SEED/pt/hf_conv/$MODEL_ID"
FINETUNE_SAVE_DIR_1="$SAVE_DIR_PREFIX/seed_$SEED/ft/${MODEL_CONFIG}_shared"
FINETUNE_SAVE_DIR_2="$SAVE_DIR_PREFIX/seed_$SEED/ft/$MODEL_CONFIG"
RESULTS_DIR="$SAVE_DIR_PREFIX/seed_$SEED/results"

#######################################################################
# INIT LOGGING
#######################################################################

SLURM_DEFAULT_FILE_PATTERN="%j.out"
SLURM_LOG_ARGS="-o $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN -e $LOG_DIR/$SLURM_DEFAULT_FILE_PATTERN"

if [[ ! -d "$PRETRAIN_SAVE_DIR" ]]; then
    mkdir -p "$PRETRAIN_SAVE_DIR" "$CONVERT_SAVE_DIR" "$FINETUNE_SAVE_DIR_1" "$FINETUNE_SAVE_DIR_2" "$LOG_DIR" "$RESULTS_DIR"
elif [[ "$FORCE" == 1 ]]; then
    echo "Overwriting existing directory $PRETRAIN_SAVE_DIR" | tee -a "$LOG_DIR/MAIN"
    rm -rf "$PRETRAIN_SAVE_DIR" && mkdir -p "$PRETRAIN_SAVE_DIR"
    rm -rf "$CONVERT_SAVE_DIR" && mkdir -p "$CONVERT_SAVE_DIR"
    rm -rf "$FINETUNE_SAVE_DIR_1" && mkdir -p "$FINETUNE_SAVE_DIR_1"
    rm -rf "$FINETUNE_SAVE_DIR_2" && mkdir -p "$FINETUNE_SAVE_DIR_2"
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
    $BASE/pretraining/pretrain_roberta_fairseq.sh \
    --data_dir "$DATA_DIR" \
    --save_dir "$PRETRAIN_SAVE_DIR" \
    --model_config "$MODEL_CONFIG" \
    --task "$TASK" \
    --seed "$SEED"
    )

echo "  id_pretrain: $id_pretrain | $LOG_DIR/$id_pretrain.out" | tee -a "$LOG_DIR/MAIN"

# run fs -> hf conversion
id_convert=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_pretrain \
    $SLURM_ARGS_GENERIC \
    $SLURM_LOG_ARGS \
    $BASE/pretraining/convert_roberta.sh \
    $PRETRAIN_SAVE_DIR \
    $DATA_DIR \
    $DATA_DIR/../tok/tokenizer \
    $CONVERT_SAVE_DIR
)

echo "  id_convert: $id_convert | $LOG_DIR/$id_convert.out" | tee -a "$LOG_DIR/MAIN"

# finetune and generate with tied encoder-decoder weights (only difference = --tie_encoder_decoder True)
id_finetune_1=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_convert \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$CONVERT_SAVE_DIR" -o "$FINETUNE_SAVE_DIR_1" -s "$SEED" --is_encoder_decoder True --tie_encoder_decoder True
)

echo "  id_finetune_1: $id_finetune_1 | $LOG_DIR/$id_finetune_1.out" | tee -a "$LOG_DIR/MAIN"

# run generation/evaluation
id_generate_1=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_finetune_1 \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_generation_exp_parallel.sh \
    -m "$FINETUNE_SAVE_DIR_1" -b 120 -o "$RESULTS_DIR"
)

echo "  id_generate_1: $id_generate_1 | $LOG_DIR/$id_generate_1.out" | tee -a "$LOG_DIR/MAIN"

# finetune and generate without tied encoder-decoder weights  (only difference = --tie_encoder_decoder False)
id_finetune_2=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_convert \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_finetuning.sh \
    -i "$CONVERT_SAVE_DIR" -o "$FINETUNE_SAVE_DIR_2" -s "$SEED" --is_encoder_decoder True --tie_encoder_decoder False
)

echo "  id_finetune_2: $id_finetune_2 | $LOG_DIR/$id_finetune_2.out" | tee -a "$LOG_DIR/MAIN"

id_generate_2=$(
    $BASE/jobs/sbatch_bare.sh \
    --dependency=afterok:$id_finetune_2 \
    $SLURM_ARGS_VOLTA \
    $SLURM_LOG_ARGS \
    $BASE/jobs/run_generation_exp_parallel.sh \
    -m "$FINETUNE_SAVE_DIR_2" -b 120 -o "$RESULTS_DIR"
)

echo "  id_generate: $id_generate_2 | $LOG_DIR/$id_generate_2.out" | tee -a "$LOG_DIR/MAIN"