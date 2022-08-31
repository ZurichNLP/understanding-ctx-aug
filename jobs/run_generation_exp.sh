#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:Tesla-V100-32GB:1
#SBATCH --partition=volta
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_generation_exp.sh  -m resources/models/ft/bart_small-rl1_mr01_rt1_ps1_in0_pl3_ma03 -e xa_knowledge

# NOTE, for smaller models, you can parallelise the experiments with jobs/run_generation_exp_parallel.sh!
# For larger models, parallelisation fails.
# BART-base and T5-small generation experiments can be run on the volta partition with --gres=gpu:1
# RoBERTA-base requires a larger GPU, e.g. --gres=gpu:Tesla-V100-32GB:1
# Alternatively, you could use reduce the batch size used for generation...

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/net/cephfs/data/tkew/projects/unsup_cntrl'
batch_size=120

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -r repo_base -m model_path [-e exp_id]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:m:e:b:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    m) model_path="$OPTARG" ;;
    e) exp_id="$OPTARG" ;;
    b) batch_size="$OPTARG" ;;
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

# cd to base dir
cd "$repo_base" && echo $(pwd) || exit 1

#######################################################################
# ACTIVATE ENV
#######################################################################

source start.sh

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

if [[ -z $exp_id ]]; then
    for exp_id in "baseline" "xa_knowledge" "xa_dialog" "qu_ctxt_aug1" "qu_ctxt_aug5" "xa_knowledge+qu_ctxt_aug5" "xa_dialog+qu_ctxt_aug5"; do
        echo "Running experiment $exp_id"
        python generation_exp.py --model_dir "$model_path"  --batch_size $batch_size --exp_id "$exp_id"
    done
else
    python generation_exp.py --model_dir "$model_path" --batch_size $batch_size --exp_id "$exp_id"
fi

echo ""
echo "Done."
echo ""