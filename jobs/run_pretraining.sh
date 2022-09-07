#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=%j.out

# Author: T. Kew
#  sbatch jobs/run_pretraining.sh -p sm_baseline [-s seed]

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

BASE='/net/cephfs/data/tkew/projects/unsup_cntrl'
SEED=4

# argument parser
while [[ $# -gt 0 ]]; do
    case $1 in
        --base)
            BASE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2" # "resources/data/books1/bin"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --replace_length)
            REPLACE_LENGTH="$2"
            shift 2 # past argument
            ;;
        --mask_random)
            MASK_RANDOM="$2"
            shift 2 # past argument
            ;;
        --rotate)
            ROTATE="$2"
            shift 2
            ;;
        --permute_sentences)
            PERMUTE_SENTENCES="$2"
            shift 2 # past argument
            ;;
        --insert)
            INSERT="$2"
            shift 2 # past argument
            ;;
        --poisson_lambda)
            POISSON_LAMBDA="$2"
            shift 2 # past argument
            ;;
        --mask)
            MASK="$2"
            shift 2 # past argument
            ;; 
        -*|--*)
            echo "Unknown option $1" && exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done


# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "see list of args in $script"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# checking required arguments
if [[ -z $BASE ]]; then
    print_missing_arg "[-r BASE]" "Base directory of the repository" && print_usage && exit 1
fi

if [[ -z $DATA_DIR ]]; then
    print_missing_arg "[--data_dir]" "path to data directory" && print_usage && exit 1
fi

if [[ -z $SAVE_DIR ]]; then
    print_missing_arg "[--save_dir]" "path to save directory" && print_usage && exit 1
fi

if [[ -z $MODEL_CONFIG ]]; then
    print_missing_arg "[--model_config]" "model config (e.g. bart_small)" && print_usage && exit 1
fi

if [[ -z $TASK ]]; then
    print_missing_arg "[--task]" "fairseq task (e.g. denoising)" && print_usage && exit 1
fi

if [[ -z $SEED ]]; then
    print_missing_arg "[--seed]" "random seed" && print_usage && exit 1
fi

# BART specific args
if [[ -z $REPLACE_LENGTH ]]; then
    print_missing_arg "[--replace_length]" "replace length for BART pretraining" && print_usage && exit 1
fi

if [[ -z $MASK_RANDOM ]]; then
    print_missing_arg "[--mask_random]" "mask random for BART pretraining" && print_usage && exit 1
fi

if [[ -z $ROTATE ]]; then
    print_missing_arg "[--rotate]" "rotate for BART pretraining" && print_usage && exit 1
fi

if [[ -z $PERMUTE_SENTENCES ]]; then
    print_missing_arg "[--permute_sentences]" "permute sentences for BART pretraining" && print_usage && exit 1
fi

if [[ -z $INSERT ]]; then
    print_missing_arg "[--insert]" "insert for BART pretraining" && print_usage && exit 1
fi

if [[ -z $POISSON_LAMBDA ]]; then
    print_missing_arg "[--poisson_lambda]" "poisson lambda for BART pretraining" && print_usage && exit 1
fi

if [[ -z $MASK ]]; then
    print_missing_arg "[--mask]" "mask for BART pretraining" && print_usage && exit 1
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

bash pretraining/pretrain_bart_fairseq.sh \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --model_config "$MODEL_CONFIG" \
    --task "$TASK" \
    --seed "$SEED" \
    --replace_length "$REPLACE_LENGTH" \
    --mask_random "$MASK_RANDOM" \
    --rotate "$ROTATE" \
    --permute_sentences "$PERMUTE_SENTENCES" \
    --insert "$INSERT" \
    --poisson_lambda "$POISSON_LAMBDA" \
    --mask "$MASK"