#!/usr/bin/env bash
# -*- coding: utf-8 -*-

########################################################
# Example usage:
# . ./run_finetuning.sh && fine_tune_bart_base_for_kgd 3

# . ./run_finetuning.sh && fine_tune_t5_small_for_kgd 3
########################################################

dummy_finetune() {
    
    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    echo "Running on GPU(s) $GPU"

    python train.py \
        --model_name_or_path "facebook/bart-base" \
        --output_dir models/dummy \
        --overwrite_output_dir True \
        --train_file data/Topical-Chat/KGD/train.json \
        --validation_file data/Topical-Chat/KGD/valid_freq.json \
        --test_file data/Topical-Chat/KGD/test_freq.json \
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
    
    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    echo "Running on GPU(s) $GPU"

    # python train.py \
    #     --model_name_or_path "models/dummy/checkpoint-2" \
    #     --output_dir models/dummy \
    #     --test_file data/Topical-Chat/KGD/test_freq.json \
    #     --text_column "turns" \
    #     --summary_column "target" \
    #     --knowledge_column "knowledge" \
    #     --num_beams 4 \
    #     --seed 42 \
    #     --do_predict \
    #     --max_predict_samples 5 --report_to "none"

    python inference2.py \
        --model_name_or_path "models/bart-base/checkpoint-21786" \
        --output_dir models/dummy \
        --test_file data/Topical-Chat/KGD/test_freq.json \
        --text_column "turns" \
        --summary_column "target" \
        --knowledge_column "knowledge" \
        --num_beams 4 \
        --seed 42 \
        --max_predict_samples 5 \
        --cross_attention_bias 5

    # python inference.py \
    #     --model_name_or_path "models/bart-base/checkpoint-21786" \
    #     --output_dir models/dummy \
    #     --test_file data/Topical-Chat/KGD/test_freq.json \
    #     --text_column "turns" \
    #     --summary_column "target" \
    #     --knowledge_column "knowledge" \
    #     --num_beams 4 \
    #     --seed 42 \
    #     --max_predict_samples 5

}

finetune_bart_base_for_kgd() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    echo "Running on GPU(s) $GPU"

    python train.py \
        --model_name_or_path "facebook/bart-base" \
        --output_dir models/bart-base \
        --overwrite_output_dir True \
        --train_file data/Topical-Chat/KGD/train.json \
        --validation_file data/Topical-Chat/KGD/valid_freq.json \
        --test_file data/Topical-Chat/KGD/test_freq.json \
        --text_column "turns" \
        --summary_column "target" \
        --knowledge_column "knowledge" \
        --overwrite_cache True \
        --preprocessing_num_workers 16 \
        --max_target_length 64 \
        --learning_rate 0.0000625 \
        --num_beams 4 \
        --num_train_epochs 10 \
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
        --report_to "wandb"

    echo ""
    echo "Finished fine-tuning run!"
    echo ""
}

finetune_t5_small_for_kgd() {

    GPU=$1
    export CUDA_VISIBLE_DEVICES=$GPU

    echo "Running on GPU(s) $GPU"

    python train.py \
        --model_name_or_path "t5-small" \
        --output_dir models/t5-small \
        --overwrite_output_dir True \
        --train_file data/Topical-Chat/KGD/train.json \
        --validation_file data/Topical-Chat/KGD/valid_freq.json \
        --test_file data/Topical-Chat/KGD/test_freq.json \
        --text_column "turns" \
        --summary_column "target" \
        --knowledge_column "knowledge" \
        --overwrite_cache True \
        --preprocessing_num_workers 16 \
        --max_target_length 64 \
        --learning_rate 0.0000625 \
        --num_beams 4 \
        --num_train_epochs 10 \
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
        --report_to "wandb"

    echo ""
    echo "Finished fine-tuning run!"
    echo ""
}