#!/usr/bin/env bash
# -*- coding: utf-8 -*-

##################################################################################################
# Example usage:

    # bash finetune.sh model_name_or_path save_dir seed [optional args (see below)]
##################################################################################################

model_name_or_path=$1
save_dir=$2
seed=${3:-42}
is_encoder_decoder=${4:-False}
tie_encoder_decoder=${5:-False}
max_train_samples=${6:-1.0}
eval_runs_per_epoch=${7:-1} # for ablations
init_as_random=${8:-False}
data_dir=${9:-"resources/data/Topical-Chat/KGD"}
log_file=${10:-"$save_dir/finetune.log"}

# check if required args are provided
{ [ -z "$model_name_or_path" ] || [ -z "$save_dir" ]; } && echo "Usage: bash finetune.sh model_name_or_path save_dir [seed] [is_encoder_decoder] [tie_encoder_decoder]" && exit 1

# create save dir (will overwrite if exists)
rm -rf "$save_dir"
mkdir -p "$save_dir"

echo ""
echo -e "model_name_or_path:\t$model_name_or_path"
echo -e "seed:\t\t\t$seed"
echo -e "data_dir:\t\t$data_dir"
echo -e "save_dir:\t\t$save_dir"
echo -e "log_file:\t\t$log_file"
echo -e "is_encoder_decoder:\t$is_encoder_decoder"
echo -e "tie_encoder_decoder:\t$tie_encoder_decoder"
echo -e "max_train_samples:\t$max_train_samples"
echo -e "eval_runs_per_epoch:\t$eval_runs_per_epoch"
echo -e "init_as_random:\t\t$init_as_random"
echo ""

if [[ $data_dir == *"Topical-Chat"* ]]; then
    train_file="$data_dir/train.json"
    validation_file="$data_dir/valid_freq.json"
    test_file="$data_dir/test_freq.json"
    knowledge_column="knowledge"
elif [[ $data_dir == *""* ]]; then
    train_file="$data_dir/train.json"
    validation_file="$data_dir/valid.json"
    test_file="$data_dir/test.json"
    knowledge_column="context"
else
    echo "Invalid data_dir: $data_dir" && exit 1
fi

python finetune.py \
    --model_name_or_path "$model_name_or_path" \
    --output_dir "$save_dir" \
    --is_encoder_decoder "$is_encoder_decoder" \
    --tie_encoder_decoder "$tie_encoder_decoder" \
    --overwrite_output_dir True \
    --train_file "$train_file" \
    --validation_file "$validation_file" \
    --test_file "$test_file" \
    --text_column "turns" \
    --summary_column "target" \
    --knowledge_column $knowledge_column \
    --overwrite_cache True \
    --preprocessing_num_workers 1 \
    --max_source_length 256 --max_target_length 64 \
    --learning_rate 0.0000625 \
    --num_beams 4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 1 \
    --seed "$seed" \
    --fp16 \
    --do_train --do_eval \
    --save_strategy "steps" \
    --evaluation_strategy "steps" --per_device_eval_batch_size 20 \
    --eval_runs_per_epoch "$eval_runs_per_epoch" \
    --write_intermediate_eval_results False --include_inputs_for_metrics False --predict_with_generate False \
    --early_stopping False \
    --max_train_samples "$max_train_samples" \
    --load_best_model_at_end True --metric_for_best_model "loss" \
    --init_as_random "$init_as_random" \
    --report_to "wandb" | tee "$log_file"

# else

#     echo "Running with validation strategy 'epoch'"
#     echo -e "eval_runs_per_epoch:\t$eval_runs_per_epoch"
#     echo ""
#     # default fine-tuning settings
#     python finetune.py \
#         --model_name_or_path "$model_name_or_path" \
#         --output_dir "$save_dir" \
#         --is_encoder_decoder "$is_encoder_decoder" \
#         --tie_encoder_decoder "$tie_encoder_decoder" \
#         --overwrite_output_dir True \
#         --train_file "$data_dir/train.json" \
#         --validation_file "$data_dir/valid_freq.json" \
#         --test_file "$data_dir/test_freq.json" \
#         --text_column "turns" \
#         --summary_column "target" \
#         --knowledge_column "knowledge" \
#         --overwrite_cache True \
#         --preprocessing_num_workers 16 \
#         --max_source_length 256 --max_target_length 64 \
#         --learning_rate 0.0000625 \
#         --num_beams 4 \
#         --num_train_epochs 10 \
#         --per_device_train_batch_size 20 \
#         --gradient_accumulation_steps 1 \
#         --seed "$seed" \
#         --fp16 \
#         --do_train --do_eval --do_predict \
#         --save_strategy "epoch" \
#         --evaluation_strategy "epoch" --per_device_eval_batch_size 20 \
#         --eval_runs_per_epoch "$eval_runs_per_epoch" --write_intermediate_eval_results False \
#         --include_inputs_for_metrics False \
#         --early_stopping True \
#         --predict_with_generate False \
#         --max_train_samples "$max_train_samples" \
#         --load_best_model_at_end True --metric_for_best_model "loss" \
#         --report_to "wandb" | tee "$log_file"

# fi

