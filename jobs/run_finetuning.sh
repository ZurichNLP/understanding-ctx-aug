#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_finetuning.sh -m roberta-base -o resources/models/ft/roberta_base

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/net/cephfs/data/tkew/projects/unsup_cntrl'

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -r repo_base -p pretrained_model -o out_dir"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:p:o:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    p) pretrained_model="$OPTARG" ;;
    o) out_dir="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $repo_base ]]; then
    print_missing_arg "[-r repo_base]" "Base directory of the repository"
    exit 1
fi

if [[ -z $pretrained_model ]]; then
    print_missing_arg "[-p pretrained_model]" "pretrained model"
    exit 1
fi

if [[ -z $out_dir ]]; then
    print_missing_arg "[-o out_dir]" "out_dir"
    exit 1
fi

# cd to base dir
cd "$repo_base" && echo $(pwd) || exit 1

#######################################################################
# ACTIVATE ENV
#######################################################################

source start.sh

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

# import functions
source finetune.sh

echo "Launching finetuning..." && finetune_for_kgd "$pretrained_model" "$out_dir"

# case "$model_type" in
#     "roberta") 
#         echo "Launching finetuning..." && finetune_for_kgd "roberta-base" "resources/models/ft/roberta_base";;
#     "bart")
#         echo "Launching finetuning..." && finetune_for_kgd "facebook/bart-base" "resources/models/ft/bart_base";;
#     "t5") 
#          echo "Launching finetuning..." && finetune_for_kgd "t5-small" "resources/models/ft/t5_small";;
#     "dummy_bart") 
#         echo "Launching finetuning..." && dummy_finetune "facebook/bart-base" "resources/models/ft/dummy";;
#     "dummy_t5") 
#         echo "Launching finetuning..." && dummy_finetune "t5-small" "resources/models/ft/dummy";;
#     "dummy_roberta") 
#         echo "Launching finetuning..." && dummy_finetune "roberta-base" "resources/models/ft/dummy";;
#     *) 
#         echo "Model type not recognised: $model_type" && exit 1 
#         ;;
# esac

echo ""
echo "done."
echo ""