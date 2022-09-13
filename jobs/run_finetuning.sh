#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_finetuning.sh -i roberta-base -o resources/models/seed_1984/ft/roberta_base -s 1984

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

BASE='/net/cephfs/data/tkew/projects/unsup_cntrl'
SEED=42
IS_ENCODER_DECODER=False
TIE_ENCODER_DECODER=False

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -r [-b BASE] -i input_dir -o output_dir [-s seed]"
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
        -i|--input_dir)
            INPUT_DIR="$2" # "resources/data/books1/bin"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --is_encoder_decoder)
            IS_ENCODER_DECODER="$2"
            shift 2
            ;;
        --tie_encoder_decoder)
            TIE_ENCODER_DECODER="$2"
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

if [[ -z $SEED ]]; then
    print_missing_arg "[-s seed]" "random seed for finetuning run"
    exit 1
fi

if [[ -z $INPUT_DIR ]]; then
    print_missing_arg "[-i input_dir]" "pretrained model dir for finetuning (either local dir or huggingface model name)"
    exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
    print_missing_arg "[-o output_dir]" "save dir for finetuned model"
    exit 1
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

bash finetune.sh "$INPUT_DIR" "$OUTPUT_DIR" "$SEED" "$IS_ENCODER_DECODER" "$TIE_ENCODER_DECODER"
