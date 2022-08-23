#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_generation_exp.sh -m resources/models/bart-base -e xa_knowledge

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
    >&2 echo "$script -m [model_path] -e [exp_id]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "m:e:" flag; do
  case "${flag}" in
    m) model_path="$OPTARG" ;;
    e) exp_id="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $exp_id ]]; then
    print_missing_arg "[-e expirment id]" "experiment id"
    exit 1
fi
if [[ -z $model_path ]]; then
    print_missing_arg "[-m model_path]" "model"
    exit 1
fi

#######################################################################
# ACTIVATE ENV
#######################################################################

source start.sh

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

python $base/run_generation.py --model_dir "$model_path" --exp_id "$exp_id"

echo ""
echo "Done."
echo ""