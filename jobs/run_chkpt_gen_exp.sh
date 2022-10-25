#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=%j.out

# Author: T. Kew

# NOTES: 
# a single generation experiment using sampling w/ 5 seeds takes approx 5 mins on 20% of validation set
# 3 * 3 * 10 * 5 / 60 = 7.5 hours!
# a single generation experiment with beam search (k=4) takes approx 2 mins on 20% of validation set
# 3 * 3 * 10 * 2 / 60 = 3 hours
# a single generation experiment using greedy (k=1) takes approx 1 min on 20% of validation set
# 3 * 3 * 10 * 1 / 60 = 1.5 hours

# better alternative is to use the run_chkpt_gen_exp_parallel.sh script

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
    o) output_dir="$OPTARG" ;;
    d) dataset="$OPTARG" ;;
    c) checkpoint="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
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

checkpoints=$( ls -v "$model_path" | grep -P "checkpoint-" )

echo "Found checkpoints: $checkpoints"

start_main=$SECOND
for checkpoint in $checkpoints; do
    for exp_id in "baseline" "qu_ctxt_aug1" "qu_ctxt_aug5"; do
        start=$SECONDS

        echo -e "MODEL PATH:\t$model_path"    
        echo -e "EXP ID:\t\t$exp_id"    
        echo -e "CHECKPOINT:\t$checkpoint"
        echo -e "OUTPUT DIR:\t$output_dir"
        echo -e "BATCH SIZR:\t$batch_size"
        echo -e "MAX SAMPLES:\t$max_predict_samples"
        echo -e "DATASET:\t$dataset"

        python generation_exp.py \
            --model_dir "$model_path" \
            --checkpoint "$checkpoint" \
            --output_dir "$output_dir" \
            --batch_size "$batch_size" \
            --exp_id "$exp_id" \
            --dataset "$dataset" \
            --max_predict_samples "$max_predict_samples"

        duration=$(( SECONDS - start ))
        echo -e "*** Finished $checkpoint - $exp_id in $duration seconds ***"
    done
done
duration_main=$(( SECONDS - start_main ))
echo -e "*** Finished all checkpoints in $duration_main seconds ***"