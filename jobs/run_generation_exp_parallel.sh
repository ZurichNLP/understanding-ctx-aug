#!/usr/bin/env bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=lowprio
#SBATCH --array=0-7
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_generation_exp_parallel.sh -m resources/models/seed_1984/KGD/bart_mini-MLM -o resources/results/seed_1984/KGD/bart_mini-MLM -t resources/data/Topical-Chat/KGD/test_rare.json

# NOTE: the output dir is inferred from the model path and the dataset path

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/data/tkew/projects/understanding-ctx-aug'
test_file="resources/data/Topical-Chat/KGD/test_freq.json"
batch_size=120


# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -m model_path [-r repo_base] [-b batch_size] [-o output_dir] [-t test_file]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:m:b:o:t:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    # d) dataset="$OPTARG" ;;
    t) test_file="$OPTARG" ;;
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

# if [[ -z $output_dir ]]; then
#     print_missing_arg "[-o output_path]" "path for results csv"
#     exit 1
# fi
# cd to base dir
cd "$repo_base" && echo $(pwd) || exit 1

#######################################################################
# ACTIVATE ENV
#######################################################################

source $repo_base/start.sh
# source $repo_base/jobs/jobs_utils.sh


#######################################################################
# function to infer output path
#######################################################################

# these are duplicated in job_utils.sh, but importing it here causes issues with sbatch array jobs.
# get KGD, CSD, DD from a path like resources/data/Commonsense-Dialogues/CSD/...
infer_dataset_id() {
    data="$1"
    dataset_id=$(echo "$data" | cut -d'/' -f 4) # e.g. KGD, CSD, DD
    echo "$dataset_id"
}

# output path for evaluation results
infer_output_path() {
    model_path="$1"
    test_file="$2"

    # Extract the model name from the model path
    model_name=$(basename "$model_path")
    # echo "$model_name"
    if [[ "$model_name" == "bart_small"* ]]; then
        model_type="bart_small"
    else
        model_type="public_models"
    fi

    # Extract the dataset name from the dataset path
    test_file_id=$(basename "$test_file" | cut -d'.' -f1) # test_freq, test_rare or test
    dataset_id=$(infer_dataset_id $test_file)
    seed=$(echo "$model_path" | cut -d'/' -f3 | cut -d'_' -f2)

    # Construct the output path
    output_path="resources/models/seed_${seed}/${dataset_id}/results/${test_file_id}-${model_type}"

    echo "$output_path"
}

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

if [[ -z $output_dir ]]; then
    output_dir=$(infer_output_path $model_path $test_file)
    [[ -z $output_dir ]] && echo "ERROR: Could not infer output dir. Please provide one with -o" && exit 1 # exit if output dir is empty
    echo "INFERRED OUTPUT DIR:" $output_dir
fi

dataset_id=$(infer_dataset_id $test_file)

# set #SBATCH --array=0-15
# exp_ids=("baseline" "xa_knowledge" "xa_dialog" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "xa_knowledge+qu_ctxt_aug5" "xa_dialog+qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "hedging_contrast_ctxt_aug5" "hedging_evasion_ctxt_aug5" "hedging_management_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5")

# set #SBATCH --array=0-6
# exp_ids=("qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "xa_knowledge+qu_ctxt_aug5" "xa_dialog+qu_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5")

# set #SBATCH --array=0-7
# exp_ids=("baseline" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5" "excl_ctxt_aug5" "hedging_contrast_ctxt_aug5" "hedging_management_ctxt_aug5" "hedging_evasion_ctxt_aug5" "e_words_ctxt_aug5" "d_words_ctxt_aug5" "i_words_ctxt_aug5" "n_words_ctxt_aug5")

# set #SBATCH --array=0-7
exp_ids=("baseline" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "long_pos_sent_ctxt_aug5" "long_neg_sent_ctxt_aug5")

# set #SBATCH --array=0-1
# exp_ids=("long_pos_sent_ctxt_aug5" "long_neg_sent_ctxt_aug5")

export "CUDA_LAUNCH_BLOCKING"=1 

# launches a single experiment job for each exp_id in parallel
srun python generation_exp.py \
    --model_dir "$model_path" \
    --output_dir "$output_dir" \
    --dataset "$dataset_id" \
    --test_file "$test_file" \
    --batch_size "$batch_size" \
    --exp_id "${exp_ids[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Done."
echo ""
