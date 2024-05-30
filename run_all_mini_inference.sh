#!/usr/bin/env bash
# -*- coding: utf-8 -*-

### Helper script to run inference for all models on all datasets
### Usage: bash run_all_mini_inference.sh run_kgd_inference

## KGD-test_rare
run_inference() {
    model_dir=$1
    output_dir=$2
    test_set=$3

    if [ ! -d "$model_dir" ]; then
        echo "Model not found: $model_dir"
    elif [ -d "$output_dir" ]; then
        # check if all .csv files contain 6 lines
        if [ ! $(find $output_dir -name "*.csv" | xargs wc -l |  awk '{if (!/total/) print $1}' | grep -v 6 | wc -l) -eq 0 ]; then
            echo "Output directory exists but some files are incomplete: $output_dir"
            echo "Will re-run inference for $model_dir ..."
        else
            echo "Will Skip inference for $model_dir ... Output directory already exists: $output_dir"
            return
        fi
    else
        echo "Will run inference for $model_dir ..."
    fi

    echo "Running inference for $model_dir"
    sbatch jobs/run_generation_exp_parallel.sh \
        -m "${model_dir}" \
        -o "${output_dir}" \
        -t "${test_set}"
}

run_kgd_freq_inference() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do
            model_dir="resources/models/seed_${seed}/KGD/${ft_model}"
            output_dir="resources/results/seed_${seed}/KGD-test_freq/${ft_model}"
            test_set="resources/data/Topical-Chat/KGD/test_freq.json"
            run_inference "$model_dir" "$output_dir" "$test_set"
        done
    done
}

run_tc_inference() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do
            model_dir="resources/models/seed_${seed}/TC/${ft_model}"
            output_dir="resources/results/seed_${seed}/TC-test_freq/${ft_model}"
            test_set="resources/data/Topical-Chat/TC/test_freq.json"
            run_inference "$model_dir" "$output_dir" "$test_set"
        done
    done
}

run_kgd_rare_inference() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do
            model_dir="resources/models/seed_${seed}/KGD/${ft_model}"
            output_dir="resources/results/seed_${seed}/KGD-test_rare/${ft_model}"
            test_set="resources/data/Topical-Chat/KGD/test_rare.json"
            run_inference "$model_dir" "$output_dir" "$test_set"
        done
    done
}


run_tc_freq_inference() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do
            model_dir="resources/models/seed_${seed}/TC/${ft_model}"
            output_dir="resources/results/seed_${seed}/TC-test_rare/${ft_model}"
            test_set="resources/data/Topical-Chat/TC/test_rare.json"
            run_inference "$model_dir" "$output_dir" "$test_set"
        done
    done
}


run_csd_inference() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do

            model_dir="resources/models/seed_${seed}/CSD/${ft_model}"
            output_dir="resources/results/seed_${seed}/CSD/${ft_model}"
            test_set="resources/data/Commonsense-Dialogues/CSD/test.json"
            run_inference "$model_dir" "$output_dir" "$test_set"
        done
    done
}

run_dd_inference() {

    for ft_model in "bart_mini-rndm" "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
        for seed in 23 42 1984; do

            model_dir="resources/models/seed_${seed}/DD/${ft_model}"
            output_dir="resources/results/seed_${seed}/DD/${ft_model}"
            test_set="resources/data/Daily-Dialog/DD/test.json"
            run_inference "$model_dir" "$output_dir" "$test_set"
        done
    done
}

"$@"