#!/usr/bin/env bash
# -*- coding: utf-8 -*-

##################################################################################################
# Example usage:

    # bash finetune.sh model_name_or_path save_dir seed [is_encoder_decoder] [tie_encoder_decoder]
##################################################################################################

model_name_or_path=$1
save_dir=$2
seed=$3
is_encoder_decoder=${4:-False}
tie_encoder_decoder=${5:-False}
max_train_samples=${6:-1.0}
eval_runs_per_epoch=${7:-1}

data_dir="resources/data/Topical-Chat/KGD"
log_file="$save_dir/finetune.log"

# check if required args are provided
{ [ -z "$model_name_or_path" ] || [ -z "$save_dir" ]; } && echo "Usage: bash finetune.sh model_name_or_path save_dir [seed] [is_encoder_decoder] [tie_encoder_decoder]" && exit 1
# set seed if not provided
{ [ -z "$seed" ]; } && echo "Seed not provided. Using default seed 42" && seed=42

mkdir -p "$save_dir"

echo ""
echo -e "model_name_or_path:\t$model_name_or_path"
echo ""
echo -e "seed:\t\t\t$seed"
echo -e "data_dir:\t\t$data_dir"
echo -e "save_dir:\t\t$save_dir"
echo -e "log_file:\t\t$log_file"
echo ""

if [[ eval_runs_per_epoch -gt 1 ]]; then

    echo "Running with validation strategy 'steps'"
    echo -e "eval_runs_per_epoch:\t$eval_runs_per_epoch"
    echo ""
    # similar settings as below except for validation evaluation runs, 
    # where we evaluate more frequently and get more detailed metrics
    python finetune.py \
        --model_name_or_path "$model_name_or_path" \
        --output_dir "$save_dir" \
        --is_encoder_decoder "$is_encoder_decoder" \
        --tie_encoder_decoder "$tie_encoder_decoder" \
        --overwrite_output_dir True \
        --train_file "$data_dir/train.json" \
        --validation_file "$data_dir/valid_freq.json" \
        --test_file "$data_dir/test_freq.json" \
        --text_column "turns" \
        --summary_column "target" \
        --knowledge_column "knowledge" \
        --overwrite_cache True \
        --preprocessing_num_workers 16 \
        --max_source_length 256 --max_target_length 64 \
        --learning_rate 0.0000625 \
        --num_beams 4 \
        --num_train_epochs 10 \
        --per_device_train_batch_size 20 \
        --gradient_accumulation_steps 1 \
        --seed "$seed" \
        --fp16 \
        --do_train --do_eval --do_predict \
        --save_strategy "steps" --save_total_limit 1 \
        --evaluation_strategy "steps" --per_device_eval_batch_size 120 \
        --eval_runs_per_epoch "$eval_runs_per_epoch" --write_intermediate_eval_results True \
        --include_inputs_for_metrics True \
        --early_stopping False \
        --predict_with_generate True \
        --max_train_samples "$max_train_samples" \
        --load_best_model_at_end False --metric_for_best_model "loss" \
        --report_to "wandb" | tee "$log_file"

else

    echo "Running with validation strategy 'epoch'"
    echo -e "eval_runs_per_epoch:\t$eval_runs_per_epoch"
    echo ""
    # default fine-tuning settings
    python finetune.py \
        --model_name_or_path "$model_name_or_path" \
        --output_dir "$save_dir" \
        --is_encoder_decoder "$is_encoder_decoder" \
        --tie_encoder_decoder "$tie_encoder_decoder" \
        --overwrite_output_dir True \
        --train_file "$data_dir/train.json" \
        --validation_file "$data_dir/valid_freq.json" \
        --test_file "$data_dir/test_freq.json" \
        --text_column "turns" \
        --summary_column "target" \
        --knowledge_column "knowledge" \
        --overwrite_cache True \
        --preprocessing_num_workers 16 \
        --max_source_length 256 --max_target_length 64 \
        --learning_rate 0.0000625 \
        --num_beams 4 \
        --num_train_epochs 10 \
        --per_device_train_batch_size 20 \
        --gradient_accumulation_steps 1 \
        --seed "$seed" \
        --fp16 \
        --do_train --do_eval --do_predict \
        --save_strategy "epoch" --save_total_limit 1 \
        --evaluation_strategy "epoch" --per_device_eval_batch_size 120 \
        --eval_runs_per_epoch "$eval_runs_per_epoch" --write_intermediate_eval_results False \
        --include_inputs_for_metrics False \
        --early_stopping True \
        --predict_with_generate True \
        --max_train_samples "$max_train_samples" \
        --load_best_model_at_end True --metric_for_best_model "loss" \
        --report_to "wandb" | tee "$log_file"

fi

