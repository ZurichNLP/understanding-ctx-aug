#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# taken from:
# https://hpc.nih.gov/docs/job_dependencies.html via https://github.com/ZurichNLP/easier-pose-translation/blob/main/scripts/running/sbatch_bare.sh

sbr="$(/cluster/slurm-20-11-8-1/bin/sbatch "$@")"

if [[ "$sbr" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    exit 0
else
    echo "sbatch failed"
    exit 1
fi