#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=%j.out

# Author: T. Kew
#  sbatch jobs/run_pretraining.sh -p sm_baseline [-s seed]

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/net/cephfs/data/tkew/projects/unsup_cntrl'
seed=4

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -p pretraining config  [-s seed] [-r repo_base]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:p:s:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    p) pretraining_config="$OPTARG" ;;
    s) seed="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $repo_base ]]; then
    print_missing_arg "[-r repo_base]" "Base directory of the repository"
    exit 1
fi

if [[ -z $pretraining_config ]]; then
    print_missing_arg "[-p pretraining_config]" "pretraining config"
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

source pretraining/pretrain_bart_fairseq.sh

case "$pretraining_config" in
   
    "sm_baseline") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.1 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;
    "sm_no_permute") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.1 \
            --permute-sentences 0.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3
        ;;
    "sm_no_masking") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.0 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.0
        ;;
    "sm_w_rotate") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 1.0 \
            --mask-random 0.1 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;
    "sm_no_permute_w_rotate") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 1.0 \
            --mask-random 0.1 \
            --permute-sentences 0.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;
    "sm_less_masking") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.1 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.1 
        ;;
    "sm_less_perm") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.1 \
            --permute-sentences 0.1 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;
    "sm_no_random") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.0 \
            --permute-sentences 1.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;
    "sm_no_random_no_perm") 
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.0 \
            --permute-sentences 0.0 \
            --insert 0.0 \
            --poisson-lambda 3.0 \
            --mask 0.3 
        ;;
    "sm_as_t5")
        echo "Launching pretraining..." && pretrain_bart "$seed" \
            "resources/data/books1/bin" \
            "bart_small" \
            "denoising_t5" \
            --replace-length 1 \
            --rotate 0.0 \
            --mask-random 0.0 \
            --permute-sentences 0.0 \
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