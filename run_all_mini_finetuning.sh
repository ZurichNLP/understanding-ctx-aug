#!/usr/bin/env bash
# -*- coding: utf-8 -*-


for ft_model in "bart_mini-MLM_PS" "bart_mini-MLM" "bart_mini-PS" "bart_mini-SI_bart" "bart_mini-SI_mass" "bart_mini-SI_t5"; do
	for seed in 23 42 1984; do
        
        # # finetune on CSD if not already done
        # output_dir="resources/models/seed_${seed}/CSD/${ft_model}"
        # if [ ! -d "$output_dir" ]; then
        #     sbatch --wait jobs/run_finetuning.sh \
        #         -i "tannonk/${ft_model}-s${seed}" -s $seed \
        #         -o "${output_dir}" \
        #         -d "resources/data/Commonsense-Dialogues/CSD"
        # else
        #     echo "Model already exists: $output_dir"
        # fi

        # # finetune on DD if not already done
        # output_dir="resources/models/seed_${seed}/DD/${ft_model}"
        # if [ ! -d "$output_dir" ]; then
        #     sbatch --wait jobs/run_finetuning.sh \
        #         -i "tannonk/${ft_model}-s${seed}" -s $seed \
        #         -o "${output_dir}" \
        #         -d "resources/data/Daily-Dialog/DD"
        # else
        #     echo "Model already exists: $output_dir"
        # fi

        # finetune on TC if not already done
        output_dir="resources/models/seed_${seed}/TC/${ft_model}"
        if [ ! -d "$output_dir" ]; then
            sbatch jobs/run_finetuning.sh \
                -i "tannonk/${ft_model}-s${seed}" -s $seed \
                -o "${output_dir}" \
                -d "resources/data/Topical-Chat/TC"
        else
            echo "Model already exists: $output_dir"
        fi

    done
done

# random inits - take the config from the MLM_PS model, but we init the weights randomly
for seed in 23 42 1984; do
    
    # # finetune on CSD if not already done    
    # output_dir="resources/models/seed_${seed}/CSD/bart_mini-rndm"
    # if [ ! -d "$output_dir" ]; then
    #     sbatch --wait jobs/run_finetuning.sh \
    #         -i "tannonk/bart_mini-MLM_PS-s${seed}" -s $seed \
    #         -o "${output_dir}" \
    #         -d "resources/data/Commonsense-Dialogues/CSD" \
    #         --init_as_random True
    # else
    #     echo "Model already exists: $output_dir"
    # fi

    # # finetune on DD if not already done
    # output_dir="resources/models/seed_${seed}/DD/bart_mini-rndm"
    # if [ ! -d "$output_dir" ]; then
    #     sbatch --wait jobs/run_finetuning.sh \
    #         -i "tannonk/bart_mini-MLM_PS-s${seed}" -s $seed \
    #         -o "${output_dir}" \
    #         -d "resources/data/Daily-Dialog/DD" \
    #         --init_as_random True
    # else
    #     echo "Model already exists: $output_dir"
    # fi

    output_dir="resources/models/seed_${seed}/TC/bart_mini-rndm"
    if [ ! -d "$output_dir" ]; then
        sbatch jobs/run_finetuning.sh \
            -i "tannonk/bart_mini-MLM_PS-s${seed}" -s $seed \
            -o "${output_dir}" \
            -d "resources/data/Topical-Chat/TC" \
            --init_as_random True
    else
        echo "Model already exists: $output_dir"
    fi

done