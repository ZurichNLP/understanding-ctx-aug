#!/usr/bin/env bash
# -*- coding: utf-8 -*-

########################################################
# Example usage:
# . ./run_finetuning.sh && fine_tune_bart_base_for_kgd 3

# . ./run_finetuning.sh && fine_tune_t5_small_for_kgd 3
########################################################

dummy_finetune() {
    
    model_name_or_path=$1
    save_dir=$2

    { [ -z "$model_name_or_path" ] || [ -z "$save_dir" ]; } && echo "Usage: finetune_for_kgd model_name_or_path save_dir" && exit 1

    python finetune.py \
        --model_name_or_path "$model_name_or_path" \
        --output_dir "$save_dir" \
        --overwrite_output_dir True \
        --train_file resources/data/Topical-Chat/KGD/train.json \
        --validation_file resources/data/Topical-Chat/KGD/valid_freq.json \
        --test_file resources/data/Topical-Chat/KGD/test_freq.json \
        --text_column "turns" \
        --summary_column "target" \
        --knowledge_column "knowledge" \
        --overwrite_cache True \
        --preprocessing_num_workers 1 \
        --max_target_length 64 \
        --learning_rate 0.0000625 \
        --num_beams 4 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 20 \
        --gradient_accumulation_steps 1 \
        --seed 42 \
        --fp16 \
        --do_train --do_eval --do_predict \
        --evaluation_strategy "epoch" --save_strategy "epoch" \
        --save_total_limit 1 \
        --load_best_model_at_end True \
        --metric_for_best_model "loss" \
        --predict_with_generate True \
        --early_stopping_patience 1 --early_stopping_threshold 0.1 \
        --persist_datasets \
        --max_train_samples 20 --max_eval_samples 5 --max_predict_samples 5 --report_to "none"

}

dummy_inference() {
    
    # GPU=$1
    # export CUDA_VISIBLE_DEVICES=$GPU

    # echo "Running on GPU(s) $GPU"

    # python finetune.py \
    #     --model_name_or_path "resources/models/dummy/checkpoint-2" \
    #     --output_dir resources/models/dummy \
    #     --test_file resources/data/Topical-Chat/KGD/test_freq.json \
    #     --text_column "turns" \
    #     --summary_column "target" \
    #     --knowledge_column "knowledge" \
    #     --num_beams 4 \
    #     --seed 42 \
    #     --do_predict \
    #     --max_predict_samples 5 --report_to "none"

    python inference2.py \
        --model_name_or_path "resources/models/bart-base/checkpoint-21786" \
        --output_dir resources/models/dummy \
        --test_file resources/data/Topical-Chat/KGD/test_freq.json \
        --text_column "turns" \
        --summary_column "target" \
        --knowledge_column "knowledge" \
        --num_beams 4 \
        --seed 42 \
        --max_predict_samples 5 \
        --cross_attention_bias 5

    # python inference.py \
    #     --model_name_or_path "resources/models/bart-base/checkpoint-21786" \
    #     --output_dir resources/models/dummy \
    #     --test_file resources/data/Topical-Chat/KGD/test_freq.json \
    #     --text_column "turns" \
    #     --summary_column "target" \
    #     --knowledge_column "knowledge" \
    #     --num_beams 4 \
    #     --seed 42 \
    #     --max_predict_samples 5

}

finetune_for_kgd() {

    model_name_or_path=$1
    save_dir=$2
    seed=$3
    data_dir="resources/data/Topical-Chat/KGD"
    log_file=$save_dir/finetune.log

    # check if required args are provided
    { [ -z "$model_name_or_path" ] || [ -z "$save_dir" ]; } && echo "Usage: finetune_for_kgd model_name_or_path save_dir [seed]" && exit 1
    # set seed if not provided
    { [ -z "$seed" ]; } && echo "Seed not provided. Using default seed 42" && seed=42
    
    mkdir -p "$save_dir"
    echo ""
    echo -e "model_name_or_path:\t\t$model_name_or_path"
    echo ""
    echo -e "seed:\t\t\t$seed"
    echo -e "data_dir:\t\t\t$data_dir"
    echo -e "save_dir:\t\t$save_dir"
    echo -e "log_file:\t\t$log_file"
    echo ""

    python finetune.py \
        --model_name_or_path "$model_name_or_path" \
        --output_dir "$save_dir" \
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
        --evaluation_strategy "epoch" --save_strategy "epoch" \
        --save_total_limit 1 \
        --load_best_model_at_end True \
        --metric_for_best_model "loss" \
        --predict_with_generate True \
        --report_to "wandb" | tee "$log_file"

}

# model_name_or_path=$1
# save_dir=$2

# { [ -z "$model_name_or_path" ] || [ -z "$save_dir" ]; } && echo "Usage: finetune.sh model_name_or_path save_dir" && exit 1

# mkdir -p "$save_dir"
# echo "Finetuning model $model_name_or_path ..."
# echo "Output directory: $save_dir"

# python finetune.py \
#     --model_name_or_path "$model_name_or_path" \
#     --output_dir "$save_dir" \
#     --overwrite_output_dir True \
#     --train_file resources/data/Topical-Chat/KGD/train.json \
#     --validation_file resources/data/Topical-Chat/KGD/valid_freq.json \
#     --test_file resources/data/Topical-Chat/KGD/test_freq.json \
#     --text_column "turns" \
#     --summary_column "target" \
#     --knowledge_column "knowledge" \
#     --overwrite_cache True \
#     --preprocessing_num_workers 16 \
#     --max_source_length 256 --max_target_length 64 \
#     --learning_rate 0.0000625 \
#     --num_beams 4 \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 20 \
#     --gradient_accumulation_steps 1 \
#     --seed 42 \
#     --fp16 \
#     --do_train --do_eval --do_predict \
#     --evaluation_strategy "epoch" --save_strategy "epoch" \
#     --save_total_limit 1 \
#     --load_best_model_at_end True \
#     --metric_for_best_model "loss" \
#     --predict_with_generate True \
#     --report_to "wandb"

# # echo ""
# # echo "Finished fine-tuning!"
# # echo ""
