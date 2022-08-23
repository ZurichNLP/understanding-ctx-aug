#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=generic

# Author: T. Kew
# sbatch jobs/run_data_prep.sh -d resources/data/books1

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
    >&2 echo "$script -d [data_dir]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "d:" flag; do
  case "${flag}" in
    d) data_dir="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $data_dir ]]; then
    print_missing_arg "[-d data_dir]"
    exit 1
fi

#######################################################################
# ACTIVATE ENV
#######################################################################

source "$base/start.sh"
echo "CONDA ENV: $CONDA_DEFAULT_ENV"

#######################################################################
# LAUNCH JOB
#######################################################################

bash $base/pretraining/data_prep/prepare_bookcorpus.sh -d "$data_dir"
