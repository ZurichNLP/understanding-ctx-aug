#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=/data/tkew/projects/pretraining_v1/job_logs/%j.out

# Author: T. Kew
# sbatch jobs/run_pretraining.sh -p sm_baseline

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
    >&2 echo "$script -p pretraining config"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "p:" flag; do
  case "${flag}" in
    p) pretraining_config="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $pretraining_config ]]; then
    print_missing_arg "[-p pretraining_config]" "pretraining config"
    exit 1
fi

#######################################################################
# ACTIVATE ENV
#######################################################################

source "$base/start.sh"
echo "CONDA ENV: $CONDA_DEFAULT_ENV"

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

source "$base/pretraining/pretrain_bart_fairseq.sh"

case "$pretraining_config" in
   
    "sm_baseline") 
        echo "Launching pretraining..." && pretrain_bart \
            "resources/data/books1/bin" \
            bart_small \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.1 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;
    "sm_no_permute") 
        echo "Launching pretraining..." && pretrain_bart \
            "resources/data/books1/bin" \
            bart_small \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.1 \
            --permute-sentences 0.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3
        ;;
    "sm_no_masking") 
        echo "Launching pretraining..." && pretrain_bart \
            "resources/data/books1/bin" \
            bart_small \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.0 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.0
        ;;
    "sm_w_rotate") 
        echo "Launching pretraining..." && pretrain_bart \
            "resources/data/books1/bin" \
            bart_small \
            --replace-length 1 \
            --rotate 1.0 \
            --mask-random 0.1 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;

    *) 
        echo "Pretraining setting not recognised: $pretraining_config" && exit 1 
        ;;
esac

echo ""
echo "Done."
echo ""