#!/usr/bin/env bash
# -*- coding: utf-8 -*-

##################################################################################################
# Example usage:

    # bash finetune_qa.sh model_name_or_path save_dir seed 
    # e.g. bash finetune_qa.sh resources/models/seed_23/pt/hf_conv/bart_small-SI_bart/ resources/models/seed_23/ft/bart_small-SI_bart-squad/ 23 squad
##################################################################################################

model_name_or_path=$1
save_dir=$2
seed=$3
dataset=$4
enc_only=$5

log_file="$save_dir/finetune.log"

# check if required args are provided
{ [ -z "$model_name_or_path" ] || [ -z "$save_dir" ]; } && echo "Usage: bash finetune_qa.sh model_name_or_path save_dir [seed]" && exit 1
# set seed if not provided
{ [ -z "$seed" ]; } && echo "Seed not provided. Using default seed 42" && seed=42

{ [ -z "$dataset" ]; } && echo "Dataset not provided. Using default dataset squad" && dataset="squad"

mkdir -p "$save_dir"

echo ""
echo -e "model_name_or_path:\t$model_name_or_path"
echo ""
echo -e "seed:\t\t\t$seed"
echo -e "dataset:\t\t$dataset"
echo -e "save_dir:\t\t$save_dir"
echo -e "log_file:\t\t$log_file"
echo -e "enc_only:\t\t$enc_only"
echo ""

if [[ "$dataset" == "squad" ]]; then
    
    echo "Finetuning and evaluating on SQuAD..."
    python src/transformers/examples/pytorch/question-answering/run_qa.py \
        --model_name_or_path "$model_name_or_path" \
        --output_dir "$save_dir" \
        --overwrite_output_dir True \
        --dataset_name "$dataset" \
        --seed $seed \
        --overwrite_cache True \
        --preprocessing_num_workers 4 \
        --do_train --do_eval \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 3 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --max_seq_length 256 \
        --doc_stride 64 \
        --save_total_limit 1 \
        --use_encoder_only "$enc_only" \
        --report_to "wandb" | tee "$log_file"

elif [[ "$dataset" == "squad_v2" ]]; then

    echo "Finetuning and evaluating on SQuAD_v2..."
    python src/transformers/examples/pytorch/question-answering/run_qa.py \
        --model_name_or_path "$model_name_or_path" \
        --output_dir "$save_dir" \
        --overwrite_output_dir True \
        --dataset_name "$dataset" --version_2_with_negative \
        --seed $seed \
        --overwrite_cache True \
        --preprocessing_num_workers 4 \
        --do_train --do_eval \
        --per_device_train_batch_size 12 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --max_seq_length 256 \
        --doc_stride 64 \
        --save_total_limit 1 \
        --use_encoder_only "$enc_only" \
        --report_to "wandb" | tee "$log_file"

else
    echo "Dataset not supported. Exiting..." && exit 1

fi


