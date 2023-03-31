#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=lowprio
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_post_hoc_eval.sh -m resources/models/seed_1984/ft/bert_base/

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/data/tkew/projects/unsup_ctrl/'
test_set='resources/data/Topical-Chat/KGD/test_freq.json'
output_dir=''

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -r repo_base -m model_path [-t test_set] [-o output_dir]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:m:t:o:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    t) test_set="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $repo_base ]]; then
    print_missing_arg "[-r repo_base]" "repo base" && exit 1
fi

if [[ -z $model_path ]]; then
    print_missing_arg "-m model_path" "model" && exit 1
else
    model_outputs="$(find "$model_path" -type d -name outputs)"
    echo "Found model outputs: $model_outputs"
fi

if [[ -z $model_outputs ]]; then
    echo "Failed to located outputs dir in $model_path" && exit 1
fi

if [[ -z $test_set ]]; then
    print_missing_arg "[-t test_set]" "test set" && exit 1
fi

if [[ -z $output_dir ]]; then
    print_missing_arg "-o output_dir" "model" && exit 1
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

log_dir="$model_path/../../logs"
log_file="$log_dir/$(basename "$model_path")_post_hoc_eval.log"

echo "Logging to $log_file"

python evaluation/evaluation.py \
    "$model_outputs" \
    --references_file "$test_set" \
    --output_dir "$output_dir" | tee "$log_file"