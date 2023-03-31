#!/usr/bin/env bash
# -*- coding: utf-8 -*-

parse_yaml() {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
          -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
          -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

# argument parser
parse_denoising_args() {

    while [[ $# -gt 0 ]]; do
        case $1 in
            --replace-length) # fairseq uses '-' instead of '_' in the argument name
                REPLACE_LENGTH="$2"
                shift 2
                ;;
            --mask-random) # fairseq uses '-' instead of '_' in the argument name
                MASK_RANDOM="$2"
                shift 2
                ;;
            --rotate)
                ROTATE="$2"
                shift 2
                ;;
            --permute-sentences) # fairseq uses '-' instead of '_' in the argument name
                PERMUTE_SENTENCES="$2"
                shift 2
                ;;
            --insert)
                INSERT="$2"
                shift 2
                ;;
            --poisson-lambda) # fairseq uses '-' instead of '_' in the argument name
                POISSON_LAMBDA="$2"
                shift 2
                ;;
            --mask)
                MASK="$2"
                shift 2
                ;;
            --denoising-method)
                DENOISING_METHOD="$2"
                shift 2
                ;;
            -*|--*)
                echo "Unknown option $1"
                exit 1
                ;;
            *)
                POSITIONAL_ARGS+=("$1") # save positional arg
                shift # past argument
                ;;
        esac
    done

    echo "--replace-length=$REPLACE_LENGTH --mask-random=$MASK_RANDOM \
        --rotate=$ROTATE --permute-sentences=$PERMUTE_SENTENCES \
        --insert=$INSERT --poisson-lambda=$POISSON_LAMBDA \
        --mask=$MASK"

}

parse_denoising_args_to_string() {
    d_args=$1
    # returns a unique ID string for the denoised pretraining run (used as save_dir)
    d_args=${d_args/--replace-length=/rl}
    d_args=${d_args/-1/01} # for mass decoding which requires -1
    d_args=${d_args/--mask-random=/_mr}
    d_args=${d_args/--rotate=/_rt}
    d_args=${d_args/--permute-sentences=/_ps}
    d_args=${d_args/--insert=/_in}
    d_args=${d_args/--poisson-lambda=/_pl}
    d_args=${d_args/--mask=/_ma}
    # d_args=${d_args/--denoising-method=/_}
    # d_args=$(echo "${d_args}" | sed "s/\.0//g" | sed "s/ //g" | sed "s/\.//g")
    d_args=$(echo "${d_args}" | sed "s/ //g" )
    
    echo "$d_args"
}

infer_output_path() {
    model_path="$1"
    test_file="$2"

    # Extract the model name from the model path
    model_name=$(basename "$model_path")
    # echo "$model_name"
    if [[ "$model_name" == "bart_small"* ]]; then
        model_type="bart_small"
    else
        model_type="public_models"
    fi

    # Extract the dataset name from the dataset path
    test_file_id=$(basename "$test_file" | cut -d'.' -f1) # test_freq, test_rare or test
    dataset=$(echo "$test_file" | cut -d'/' -f 4) # KGD or CD
    seed=$(echo "$model_path" | cut -d'/' -f3 | cut -d'_' -f2)

    # Construct the output path
    output_path="resources/models/seed_${seed}/${dataset}/results/${test_file_id}-${model_type}"

    echo "$output_path"
}