#!/usr/bin/env bash
#SBATCH --time=2:00:00 # set to ~1 hour per generation setting
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=lowprio
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_generation_exp_mini.sh  -m resources/models/ft/bart_small-MLM_PS -e xa_knowledge

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
test_file="resources/data/Topical-Chat/KGD/test_freq.json"
batch_size=120

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -m model_path [-r repo_base] [-e exp_id] [-o output_dir] [-b batch_size] [-t test_file]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:m:e:b:o:t:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    e) exp_id="$OPTARG" ;;
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

dataset_id=$(infer_dataset_id $test_file)

echo "Running experiment $exp_id"
echo "Batch size: $batch_size"
python generation_exp.py --model_dir "$model_path" --batch_size "$batch_size" --output_dir "$output_dir" --exp_id "$exp_id" --test_file "$test_file" --dataset "$dataset_id"

echo ""
echo "Done."
echo ""
