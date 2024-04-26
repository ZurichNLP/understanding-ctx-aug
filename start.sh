#!/usr/bin/env bash
# -*- coding: utf-8 -*-

module purge
# module load anaconda3 v100-32g
module load anaconda3 gpu
# gpu gcc/8.5.0 cuda/11.6.2 
# print loaded modules
module list

eval "$(conda shell.bash hook)"
conda activate
conda activate unsup_ctrl
echo "CONDA ENV: $CONDA_DEFAULT_ENV"
