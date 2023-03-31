#!/usr/bin/env bash
#SBATCH --time=14:00:00 # set to ~1 hour per generation setting
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:V100:1
#SBARCH --constraint=GPUMEM32GB
#SBATCH --output=%j.out

# Author: T. Kew

# Example Call:
#       sbatch jobs/run_generation_exp.sh \
#           -m resources/models/seed_23/TC/ft/bart_base/ \
#           -d resources/data/Topical-Chat/KGD/test_freq.json

# NOTE: the output dir is inferred from the model path and the dataset path

# NOTE, for smaller models, you can parallelise the experiments with jobs/run_generation_exp_parallel.sh!
# For larger models, parallelisation fails.
# BART-base and T5-small generation experiments can be run on the volta partition with --gres=gpu:1
# RoBERTA-base requires a larger GPU, e.g. --gres=gpu:V100:1 --constraint=GPUMEM32GB
# Alternatively, you could use reduce the batch size used for generation...

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/data/tkew/projects/unsup_ctrl/'
batch_size=120

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -m model_path -t test_file -d dataset [-r repo_base] [-e exp_id] [-o output_dir] [-b batch_size]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:m:e:b:o:d:t:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    e) exp_id="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    t) test_file="$OPTARG" ;;
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

if [[ -z $test_file ]]; then
    print_missing_arg "[-t test_file]" "test file"
    exit 1
fi

if [[ -z $dataset ]]; then
    print_missing_arg "[-d dataset]" "dataset"
    exit 1
fi
# if [[ -z $output_dir ]]; then
#     print_missing_arg "[-o output_dir]" "output dir for results csv files"
#     exit 1
# fi

# cd to base dir
cd "$repo_base" && echo $(pwd) || exit 1

#######################################################################
# ACTIVATE ENV
#######################################################################

source $repo_base/start.sh
source $repo_base/jobs/job_utils.sh # for infer_output_path

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

if [[ -z $output_dir ]]; then
    output_dir=$(infer_output_path $model_path $test_file)
    [[ -z $output_dir ]] && echo "ERROR: Could not infer output dir. Please provide one with -o" && exit 1 # exit if output dir is empty
    echo "INFERRED OUTPUT DIR:" $output_dir
fi

if [[ -z $exp_id ]]; then
    # for exp_id in "baseline" "xa_knowledge" "xa_dialog" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "xa_knowledge+qu_ctxt_aug5" "xa_dialog+qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "hedging_contrast_ctxt_aug5" "hedging_evasion_ctxt_aug5" "hedging_management_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5"; do
    # for exp_id in "baseline" "xa_knowledge" "xa_dialog" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "xa_knowledge+qu_ctxt_aug5" "xa_dialog+qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "hedging_contrast_ctxt_aug5" "hedging_evasion_ctxt_aug5" "hedging_management_ctxt_aug5"; do
    # for exp_id in "baseline" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "xa_knowledge+qu_ctxt_aug5" "xa_dialog+qu_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5"; do
    # for exp_id in "baseline" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "long_pos_sent_ctxt_aug5" "long_neg_sent_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5" "excl_ctxt_aug5" "hedging_contrast_ctxt_aug5" "hedging_management_ctxt_aug5" "hedging_evasion_ctxt_aug5" "e_words_ctxt_aug5" "d_words_ctxt_aug5" "i_words_ctxt_aug5" "n_words_ctxt_aug5"; do
    
    # # SBATCH --time=14:00:00
    # for exp_id in "baseline" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "long_pos_sent_ctxt_aug5" "long_neg_sent_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5" "excl_ctxt_aug5"; do

    for exp_id in "baseline" "qu_ctxt_aug1" "qu_ctxt_aug5" "qu_ctxt_aug10_50" "qu_ctxt_aug10_100"; do

    # # SBATCH --time=3:00:00
    # for exp_id in "long_pos_sent_ctxt_aug5" "long_neg_sent_ctxt_aug5"; do

        echo "Running experiment $exp_id"
        echo "Batch size: $batch_size"
        python generation_exp.py --model_dir "$model_path" --batch_size "$batch_size" --output_dir "$output_dir" --exp_id "$exp_id" --test_file "$test_file" --dataset "$dataset"
    done
else
    echo "Running experiment $exp_id"
    echo "Batch size: $batch_size"
    python generation_exp.py --model_dir "$model_path" --batch_size "$batch_size" --output_dir "$output_dir" --exp_id "$exp_id" --test_file "$test_file" --dataset "$dataset"
fi

echo ""
echo "Done."
echo ""
