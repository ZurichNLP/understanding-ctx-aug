#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=vesta
#SBATCH --partition=volta
#SBATCH --array=0-9
#SBATCH --output=%j.out

# Author: T. Kew
# NOTE: the sbatch --array argument should correspond to the number of checkpoints to run inference with
#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/net/cephfs/data/tkew/projects/unsup_cntrl'
dataset="resources/data/Topical-Chat/KGD/valid_freq.json"
batch_size=120
max_predict_samples=0.2

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script [-r repo_base] -m model_path [-b batch_size] [-o output_dir]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:m:b:o:c:d:x:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    d) dataset="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    x) max_predict_samples="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $repo_base ]]; then
    print_missing_arg "[-r repo_base]" "repo base"
    exit 1
fi

if [[ -z $model_path ]]; then
    print_missing_arg "[-m model_path]" "model"
    exit 1
fi

if [[ -z $output_dir ]]; then
    print_missing_arg "[-o output_dir]" "output dir for results csv files"
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

mkdir -p "$output_dir"

exp_ids=("baseline" "qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "excl_ctxt_aug5" "qu_ctxt_aug1" "single_qu_ctxt_aug5")

# checkpoints=$( ls -v "$model_path" | grep -P "checkpoint-" )
# get the last checkpoint from each epoch
checkpoints=( $( ls -v "$model_path" | grep 'checkpoint-' | awk 'NR % 3 == 0' | tr '\n' ' ') )
# checkpoints=( $( ls -v "$model_path" | grep 'checkpoint-' | head -n 2 | tr '\n' ' ' ))
# mapfile -t checkpoints < <($( ls -v "$model_path" | grep 'checkpoint-' | awk 'NR % 3 == 0' ))
# checkpoints=$(find "$model_path" -iname "checkpoint-*" -type d)

echo "Running for " "${#exp_ids[@]}" " experiments: " "${exp_ids[@]} ..."
echo "Found " "${#checkpoints[@]}" " checkpoints: " "${checkpoints[@]} ..."

count=0
for exp_id in "${exp_ids[@]}"; do
    echo "$count: $exp_id"
    srun python generation_exp.py \
        --model_dir "$model_path" \
        --batch_size "$batch_size" \
        --output_dir "$output_dir" \
        --checkpoint "${checkpoints[$SLURM_ARRAY_TASK_ID]}" \
        --dataset "$dataset" \
        --max_predict_samples "$max_predict_samples" \
        --exp_id "$exp_id"
done

echo ""
echo "Done."
echo ""