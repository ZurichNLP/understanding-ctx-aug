#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=/data/tkew/projects/unsup_cntrl/job_logs/%j.out

# Author: T. Kew
# sbatch jobs/run_finetuning.sh -m roberta

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

script_dir=$( dirname -- "$( readlink -f -- "$0"; )"; )
base="$script_dir/.."
cd "$base" || exit 1

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -m [model_type]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "m:" flag; do
  case "${flag}" in
    m) model_type="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $model_type ]]; then
    print_missing_arg "[-m model_type]" "model"
    exit 1
fi

#######################################################################
# ACTIVATE ENV
#######################################################################

source $base/start.sh

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

# import functions
source $base/run_finetuning.sh

case "$model_type" in
   
    "roberta") 
        echo "Launching finetuning" && finetune_roberta_base_for_kgd
        ;;
    "bart") 
        echo "Launching finetuning" && finetune_bart_base_for_kgd
        ;;
    "t5") 
        echo "Launching finetuning" && finetune_t5_small_for_kgd
        ;;
    "dummy_bart") 
        echo "Launching finetuning" && dummy_finetune "facebook/bart-base"
        ;;
    "dummy_t5") 
        echo "Launching finetuning" && dummy_finetune "t5-small"
        ;;
    "dummy_roberta") 
        echo "Launching finetuning" && dummy_finetune "roberta-base"
        ;;
    *) 
        echo "Model type not recognised: $model_type" && exit 1 
        ;;
esac

echo ""
echo "done."
echo ""