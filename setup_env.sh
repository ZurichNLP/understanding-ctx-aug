#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# base=$(cd $(dirname -- $0); pwd)
# echo "$base"

# # if running on cluster, run
# module purge
# module load volta anaconda3 gcc/7.4.0 nvidia/cuda10.2-cudnn7.6.5
# # print loaded modules
# module list

# # make conda command available
# eval "$(conda shell.bash hook)"
# conda activate

# # create new clean environment
# conda create -n unsup_cntrl_3 python=3.8 -y
# conda activate unsup_cntrl_3 && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

# pip install -r requirements.txt

# # for finetuning data preprocessing
# python -m spacy download en_core_web_sm

# # to run notebook from a server with ipython kernels, run
# python -m ipykernel install --user --name=unsup_ctrl

#################
## alternatively:
#################

# mkdir -p $base/src

# echo "Installing transformers..."
# cd $base/src
# git clone git@github.com:tannonk/transformers.git
# cd transformers
# git checkout unsup_cntrl
# pip install -e .
# cd $base/src

# echo "Installing fairseq..."
# git clone git@github.com:tannonk/fairseq.git
# cd fairseq
# git checkout minibart
# pip install --editable ./
# cd $base/src

## test install
# python -c 'import torch; print(torch.__version__)'

# echo "Installing apex..."
# cd $base/src
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
# cd $base