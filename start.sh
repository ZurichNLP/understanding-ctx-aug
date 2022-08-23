#!/usr/bin/env bash
# -*- coding: utf-8 -*-

module purge
module load volta anaconda3 gcc/7.4.0 nvidia/cuda10.2-cudnn7.6.5
# print loaded modules
module list

eval "$(conda shell.bash hook)"
conda activate
conda activate unsup_cntrl_3
echo "CONDA ENV: $CONDA_DEFAULT_ENV"
