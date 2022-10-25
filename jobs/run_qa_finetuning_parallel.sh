#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --array=0-2
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_qa_finetuning_parallel.sh -m model_id -d dataset
# parallelises finetuning with multiple seed for a single model for a single dataset

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

# defaults
BASE='/net/cephfs/data/tkew/projects/unsup_cntrl'
SEEDS=(23 42 1984)

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -r [-b BASE] -m model_id -d dataset"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--base)
            BASE="$2"
            shift 2
            ;;
        -m|--model_id)
            MODEL_ID="$2" # "resources/data/books1/bin"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -e|--enc_only)
            ENC_ONLY="$2"
            shift 2
            ;;
        -*|--*)
            echo "Unknown option $1" && print_usage && exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done

# checking required arguments
if [[ -z $BASE ]]; then
    print_missing_arg "[-r BASE]" "Base directory of the repository"
    exit 1
fi

if [[ -z $MODEL_ID ]]; then
    print_missing_arg "[-m model_id]" "custom model (expected to be in resources/models/seed_*/pt/hf_conv/)"
    exit 1
fi

if [[ -z $DATASET ]]; then
    print_missing_arg "[-d dataset]" "save dir for finetuned model"
    exit 1
fi

if [[ -z $ENC_ONLY ]]; then
    print_missing_arg "[-e enc_only]" "..."
fi

# cd to base dir
cd "$BASE" && echo $(pwd) || exit 1

#######################################################################
# ACTIVATE ENV
#######################################################################

source start.sh

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

if [[ "$ENC_ONLY" == "True" ]]; then
    MODEL_TYPE="enc"
else
    MODEL_TYPE="enc_dec"
fi
    
# launches a single experiment job for each exp_id in parallel
srun bash finetune_qa.sh \
    "resources/models/seed_${SEEDS[$SLURM_ARRAY_TASK_ID]}/pt/hf_conv/$MODEL_ID" \
    "resources/models/seed_${SEEDS[$SLURM_ARRAY_TASK_ID]}/ft/$MODEL_ID-$MODEL_TYPE-$DATASET" \
    "${SEEDS[$SLURM_ARRAY_TASK_ID]}" \
    "$DATASET" \
    "$ENC_ONLY"