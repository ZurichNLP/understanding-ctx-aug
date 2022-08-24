#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=generic
#SBATCH --output=%j.out

# Author: T. Kew
# sbatch jobs/run_data_prep.sh -d resources/data/books1

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

repo_base='/net/cephfs/data/tkew/projects/unsup_cntrl'

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -r repo_base -d data_dir"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "r:d:" flag; do
  case "${flag}" in
    r) repo_base="$OPTARG" ;;
    d) data_dir="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $repo_base ]]; then
    print_missing_arg "[-r repo_base]" "Base directory of the repository"
    exit 1
fi

if [[ -z $data_dir ]]; then
    print_missing_arg "[-d data_dir]"
    exit 1
fi

# cd to base dir/pretraining
cd "$repo_base/pretraining" && echo $(pwd) || exit 1

#######################################################################
# ACTIVATE ENV
#######################################################################

source start.sh

#######################################################################
# LAUNCH JOB
#######################################################################

bash prepare_bookcorpus.sh -d "$data_dir"
