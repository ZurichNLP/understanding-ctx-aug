#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# example usage:
# . jobs/run_ablation_generation_exp.sh -s 23 -m bart_small-SI_t5

BASE='/net/cephfs/data/tkew/projects/unsup_cntrl'
RUN_FINETUNING="false" # for partial runs

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -m Model name/ID [-b repo base] [-s seed]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "b:s:c:m:f:" flag; do
  case "${flag}" in
    s) SEED="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    m) MODEL_ID="$OPTARG" ;;
    f) RUN_FINETUNING="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $BASE ]]; then
    print_missing_arg "[-b BASE]" "Base directory of the repository"
    exit 1
fi

if [[ -z $MODEL_ID ]]; then
    print_missing_arg "[-m MODEL_ID]" "Model name/ID"
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

# #######################################################################
# # SLURM JOB ARGS
# #######################################################################

SLURM_ARGS_VOLTA="--qos=vesta --time=10:00:00 --gres gpu:1 --cpus-per-task 1 --mem-per-cpu=8G --partition=volta"

# #######################################################################
# # SET EXPERIMENT SETTINGS 
# #######################################################################

SAVE_DIR_PREFIX="resources/models/seed_${SEED}"
# MODEL_NAME=$(basename "$MODEL_DIR") # resources/models/seed_23/ft_ablations/bart_small-SI_bart-0.1 -> bart_small-SI_bart-0.1
LOG_DIR="$SAVE_DIR_PREFIX/logs/$MODEL_ID"
MODEL_DIR="$SAVE_DIR_PREFIX/pt/hf_conv/$MODEL_ID"
BASE_OUTPUT_DIR="$SAVE_DIR_PREFIX/ft/$MODEL_ID"
RESULTS_DIR="$SAVE_DIR_PREFIX/results"

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
echo "MODEL_DIR: $MODEL_DIR" | tee -a "$LOG_DIR/MAIN"
echo "OUTPUT_DIR: $BASE_OUTPUT_DIR-X.XX" | tee -a "$LOG_DIR/MAIN"
echo "RESULTS_DIR: $RESULTS_DIR" | tee -a "$LOG_DIR/MAIN"
echo "LOG_DIR: $LOG_DIR" | tee -a "$LOG_DIR/MAIN"
echo "##############################################" | tee -a "$LOG_DIR/MAIN"


# NOTE: the call below submits a parallel job array (1 for each amount of training data)

if [[ $RUN_FINETUNING == "true" ]]; then

    echo "Running finetuning" | tee -a "$LOG_DIR/MAIN"

    id_finetune=$(
        $BASE/jobs/sbatch_bare.sh \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_finetuning_ablation.sh \
        -i "$MODEL_DIR" -o "$BASE_OUTPUT_DIR" -s "$SEED"
    )
    echo "  id_finetune: $id_finetune | $LOG_DIR/$id_finetune.out" | tee -a "$LOG_DIR/MAIN"

    # these job calls are hardcoded for now and based on train_samples=(0.1 0.25 0.5 0.75) defined in run_finetuning_ablation.sh 
    # NOTE: each call submits a parallel job array (1 for each generation experiment setting)

    id_generate_1=$(
        $BASE/jobs/sbatch_bare.sh \
        --dependency=afterok:$id_finetune \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.1" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_1: $id_generate_1 | $LOG_DIR/$id_generate_1.out" | tee -a "$LOG_DIR/MAIN"

    id_generate_2=$(
        $BASE/jobs/sbatch_bare.sh \
        --dependency=afterok:$id_finetune \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.25" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_2: $id_generate_2 | $LOG_DIR/$id_generate_2.out" | tee -a "$LOG_DIR/MAIN"

    id_generate_3=$(
        $BASE/jobs/sbatch_bare.sh \
        --dependency=afterok:$id_finetune \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.5" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_3: $id_generate_3 | $LOG_DIR/$id_generate_3.out" | tee -a "$LOG_DIR/MAIN"

    id_generate_4=$(
        $BASE/jobs/sbatch_bare.sh \
        --dependency=afterok:$id_finetune \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.75" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_4: $id_generate_4 | $LOG_DIR/$id_generate_4.out" | tee -a "$LOG_DIR/MAIN"

elif [[ $RUN_FINETUNING == "false" ]]; then

    echo "Skipping finetuning. Expecting to find models $BASE_OUTPUT_DIR-X.XX" | tee -a "$LOG_DIR/MAIN"

    # run generation with no slurm dependency
    # these job calls are hardcoded for now and based on train_samples=(0.1 0.25 0.5 0.75) defined in run_finetuning_ablation.sh 
    # NOTE: each call submits a parallel job array (1 for each generation experiment setting)

    id_generate_1=$(
        $BASE/jobs/sbatch_bare.sh \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.1" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_1: $id_generate_1 | $LOG_DIR/$id_generate_1.out" | tee -a "$LOG_DIR/MAIN"

    id_generate_2=$(
        $BASE/jobs/sbatch_bare.sh \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.25" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_2: $id_generate_2 | $LOG_DIR/$id_generate_2.out" | tee -a "$LOG_DIR/MAIN"

    id_generate_3=$(
        $BASE/jobs/sbatch_bare.sh \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.5" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_3: $id_generate_3 | $LOG_DIR/$id_generate_3.out" | tee -a "$LOG_DIR/MAIN"

    id_generate_4=$(
        $BASE/jobs/sbatch_bare.sh \
        $SLURM_ARGS_VOLTA \
        $SLURM_LOG_ARGS \
        $BASE/jobs/run_generation_exp_parallel.sh \
        -m "$BASE_OUTPUT_DIR-0.75" -b 120 -o "$RESULTS_DIR"
    )
    echo "  id_generate_4: $id_generate_4 | $LOG_DIR/$id_generate_4.out" | tee -a "$LOG_DIR/MAIN"

fi