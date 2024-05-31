#!/usr/bin/env bash
# -*- coding: utf-8 -*-

### Helper script to re-run evals for all models on all datasets
### Warning: This script will launch multiple parallel jobs on the cluster!
### Usage: bash run_all_evals.sh run_kgd_eval
### bash run_all_evals.sh run_tc_eval

## KGD-test_rare
run_eval() {
    model_dir=$1
    test_set=$2
    output_dir=$3

    if [ ! -d "$model_dir" ]; then
        echo "Model not found: $model_dir"
        exit 1
    elif [ ! -d "$output_dir" ]; then
        mkdir -p $output_dir
        echo "Will run evals for $model_dir ..."

    # not tested!
    # elif [ -d "$output_dir" ]; then
    #     # check if all .csv files contain 6 lines
    #     if [ ! $(find $output_dir -name "*.csv" | xargs wc -l |  awk '{if (!/total/) print $1}' | grep -v 6 | wc -l) -eq 0 ]; then
    #         echo "Output directory exists but some files are incomplete: $output_dir"
    #         echo "Will re-run evals for $model_dir ..."
    #     else
    #         echo "Will Skip evals for $model_dir ... Output directory already exists and files complete: $output_dir"
    #         return
    #     fi
        
    fi


    echo "Running eval for $model_dir"
    
    sbatch jobs/run_post_hoc_eval.sh \
        -m $model_dir \
        -t $test_set \
        -o $output_dir
}


run_kgd_eval_mini() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do
            for split in "test_freq" "test_rare"; do
                model_dir="resources/models/seed_${seed}/KGD/${ft_model}/outputs"
                test_set="resources/data/Topical-Chat/KGD/${split}.json"
                output_dir="resources/results_with_chrf/seed_${seed}/KGD-${split}/${ft_model}"
                run_eval "$model_dir" "$test_set" "$output_dir"
            done
        done
    done
}


run_kgd_eval_public() {

    for ft_model in "bart_base" "bart_large" "t5_small" "t5_lm_small" "t5v11_base" "t5v11_small"; do
        for seed in 23 42 1984; do
            for split in "test_freq"; do
                model_dir="resources/models/seed_${seed}/KGD/${ft_model}/outputs"
                test_set="resources/data/Topical-Chat/KGD/${split}.json"
                output_dir="resources/results_with_chrf/seed_${seed}/KGD-${split}/${ft_model}"
                run_eval "$model_dir" "$test_set" "$output_dir"
            done
        done
    done
}

run_tc_eval() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do
            for split in "test_freq"; do
                model_dir="resources/models/seed_${seed}/TC/${ft_model}"
                test_set="resources/data/Topical-Chat/TC/test_freq.json"
                output_dir="resources/results_with_chrf/seed_${seed}/TC-${split}/${ft_model}"
                run_eval "$model_dir" "$test_set" "$output_dir" 
            done
        done
    done
}


run_csd_eval() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do
            model_dir="resources/models/seed_${seed}/CSD/${ft_model}"
            test_set="resources/data/Commonsense-Dialogues/CSD/test.json"
            output_dir="resources/results_with_chrf/seed_${seed}/CSD/${ft_model}"
            run_eval "$model_dir" "$test_set" "$output_dir" 
        done
    done
}

run_dd_inference() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do

            model_dir="resources/models/seed_${seed}/DD/${ft_model}"
            test_set="resources/data/Daily-Dialog/DD/test.json"
            output_dir="resources/results_with_chrf/seed_${seed}/DD/${ft_model}"
            run_eval "$model_dir" "$test_set" "$output_dir" 
        done
    done
}

"$@"