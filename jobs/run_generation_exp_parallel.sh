#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:Tesla-V100-32GB:1
#SBATCH --partition=volta
#SBATCH --array=0-9
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_generation_exp_parallel.sh -m resources/models/ft/bart_small-rl1_mr01_rt1_ps1_in0_pl3_ma03

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/net/cephfs/data/tkew/projects/unsup_cntrl'
dataset="resources/data/Topical-Chat/KGD/test_freq.json"
batch_size=120


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
while getopts "r:m:b:o:d:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    d) dataset="$OPTARG" ;;
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
    print_missing_arg "[-o output_path]" "path for results csv"
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

exp_ids=("baseline" "xa_knowledge" "xa_dialog" "qu_ctxt_aug1" "qu_ctxt_aug5" "xa_knowledge+qu_ctxt_aug5" "xa_dialog+qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neu_sent_ctxt_aug5" "neg_sent_ctxt_aug5")

# launches a single experiment job for each exp_id in parallel
srun python generation_exp.py \
    --model_dir "$model_path" \
    --output_dir "$output_dir" \
    --dataset "$dataset" \
    --batch_size "$batch_size" \
    --exp_id "${exp_ids[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Done."
echo ""