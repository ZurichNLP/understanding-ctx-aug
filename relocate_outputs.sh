#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# models/seed_23/ft/bart_base/outputs/* outputs/seed_23/KGD/bart_base/

for seed in 23 42 1984; do
    # iterate over all model dirs
    for model_dir in "resources/models/seed_${seed}/KGD/"*; do
        
        if [[ -d $model_dir ]]; then
            model_name=$(basename $model_dir)
            echo $model_name
            # iterate over all outputs
            output_dir="resources/outputs/seed_${seed}/KGD/${model_name}"
            mkdir -p $output_dir
            
            # get number of files in output dir
            num_files=$(ls -1q $model_dir/outputs | wc -l)
            if [[ $num_files -eq 0 ]]; then
                echo "No files in $model_dir/outputs"
            else
                echo "Copying $num_files files from $model_dir/outputs to $output_dir"
            
                cp -r $model_dir/outputs/* $output_dir/

                # check if all files were copied
                num_files_copied=$(ls -1q $output_dir | wc -l)
                if [[ $num_files_copied -eq $num_files ]]; then
                    echo "All files copied successfully"
                else
                    echo "Error: $num_files_copied files copied out of $num_files"
                fi
            fi
        fi
    done
done

