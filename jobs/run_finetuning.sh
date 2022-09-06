#!/bin/bash
#SBATCH --time=00:10:00
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
seed=42

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -r repo_base -i in_dir -o out_dir"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:i:o:s:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    i) in_dir="$OPTARG" ;;
    o) out_dir="$OPTARG" ;;
    s) seed="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $repo_base ]]; then
    print_missing_arg "[-r repo_base]" "Base directory of the repository"
    exit 1
fi

if [[ -z $seed ]]; then
    print_missing_arg "[-s seed]" "random seed for finetuning run"
    exit 1
fi

if [[ -z $in_dir ]]; then
    print_missing_arg "[-i in_dir]" "pretrained model dir for finetuning"
    exit 1
fi

if [[ -z $out_dir ]]; then
    print_missing_arg "[-o out_dir]" "save dir for finetuned model"
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

echo "Launching finetuning..." && finetune_for_kgd "$in_dir" "$out_dir" "$seed"