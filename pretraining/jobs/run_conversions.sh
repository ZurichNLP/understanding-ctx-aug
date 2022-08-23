#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=generic
#SBATCH --output=/data/tkew/projects/pretraining_v1/job_logs/%j.out

# Author: T. Kew
# sbatch jobs/run_conversions.sh -i

# --checkpoint resources/models/pt/fairseq/bart_small/checkpoint_best.pt \
# --tokenizer resources/data/books1/tok/tokenizer \
# --out_dir resources/models/pt/huggingface_conv/bart_small

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
base="$script_dir/../.."
cd "$base" || exit 1

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -c checkpoint -o out_dir -t tokenizer"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "c:t:o:" flag; do
  case "${flag}" in
    c) checkpoint_dir="$OPTARG" ;;
    o) out_dir="$OPTARG" ;;
    t) tokenizer="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $checkpoint_dir ]]; then
    print_missing_arg "[-c checkpoint_dir]" "path to directory containing model (checkpoint_best.pt) trained with Fairseq"
    exit 1
fi

if [[ -z $out_dir ]]; then
    print_missing_arg "[-o out_dir]" "path to save converted modelfor Hugging Face "
    exit 1
fi

if [[ -z $tokenizer ]]; then
    print_missing_arg "[-t tokenizer]" "path to Hugging Face tokenizer used to prepare data for Fairseq"
    exit 1
fi


#######################################################################
# ACTIVATE ENV
#######################################################################

source "$base/start.sh"
echo "CONDA ENV: $CONDA_DEFAULT_ENV"

#######################################################################
# LAUNCH
#######################################################################

python $base/pretraining/convert_fairseq_model_to_transformers.py \
    --checkpoint "$checkpoint_dir/checkpoint_best.pt" \
    --tokenizer "$tokenizer" \
    --out_dir "$out_dir"