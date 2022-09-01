#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# argument parser
parse_denoising_args() {

    while [[ $# -gt 0 ]]; do
        case $1 in
            --replace-length)
                REPLACE_LENGTH="$2"
                shift # past argument
                shift # past value
                ;;
            --mask-random)
                MASK_RANDOM="$2"
                shift # past argument
                shift # past value
                ;;
            --rotate)
                ROTATE="$2"
                shift # past argument
                shift # past value
                ;;
            --permute-sentences)
                PERMUTE_SENTENCES="$2"
                shift # past argument
                shift # past value
                ;;
            --insert)
                INSERT="$2"
                shift # past argument
                shift # past value
                ;;
            --poisson-lambda)
                POISSON_LAMBDA="$2"
                shift # past argument
                shift # past value
                ;;
            --mask)
                MASK="$2"
                shift # past argument
                shift # past value
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

    echo "--replace-length=$REPLACE_LENGTH --mask-random=$MASK_RANDOM --rotate=$ROTATE --permute-sentences=$PERMUTE_SENTENCES --insert=$INSERT --poisson-lambda=$POISSON_LAMBDA --mask=$MASK"

}

get_denoising_args_string() {
    d_args=$1
    # returns an ID string for the denoised pretraining run
    d_args=${d_args/--replace-length=/rl}
    d_args=${d_args/--mask-random=/_mr}
    d_args=${d_args/--rotate=/_rt}
    d_args=${d_args/--permute-sentences=/_ps}
    d_args=${d_args/--insert=/_in}
    d_args=${d_args/--poisson-lambda=/_pl}
    d_args=${d_args/--mask=/_ma}
    d_args=$(echo "${d_args}" | sed "s/\.0//g" | sed "s/ //g" | sed "s/\.//g")
    
    echo "$d_args"
}

convert_to_hf() {

    echo "Running conversion..."

    checkpoint_dir="$1" # "resources/models/pt/fairseq/bart_small/Rl0Mr0Rt0Ps0Pl0In0Ma0"
    tokenizer_dir="$2" # "resources/data/books1/tokenizer"
    out_dir="$3" # "resources/models/pt/hf_conv/bart_small/Rl0Mr0Rt0Ps0Pl0In0Ma0"

    python convert_fairseq_model_to_transformers.py \
    --checkpoint "$checkpoint_dir/checkpoint_best.pt" \
    --tokenizer "$tokenizer_dir" \
    --out_dir "$out_dir"

    echo "$out_dir"
}

# this training command is adapted from the original provided by M. Lewis
# https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320
pretrain_bart() {

    seed="$1"
    data_dir="$2" # "resources/data/books1/bin"
    model_config="$3" # bart_small
    task="$4"

    # parse arguments for denoising objectives
    #https://gist.github.com/jimratliff/d735a57eef05b650d4a17f10b7da64d9
    read -r replace_length mask_random rotate permute_sentences insert poisson_lambda mask <<< "$(parse_denoising_args "$@")"
    
    # get an ID string for the denoised pretraining run
    denoising_args=$(get_denoising_args_string "${replace_length} ${mask_random} ${rotate} ${permute_sentences} ${insert} ${poisson_lambda} ${mask}")
    
    fairseq_save_dir="resources/models/seed_$seed/pt/fairseq/${model_config}-${task}-${denoising_args}"
    
    echo ""
    echo -e "model_config:\t\t$model_config"
    echo ""
    echo -e "seed:\t\t\t$seed"
    echo -e "task:\t\t\t$task"
    echo -e "replace-length:\t\t$replace_length"
    echo -e "mask-random:\t\t$mask_random"
    echo -e "rotate:\t\t\t$rotate"
    echo -e "permute-sentences:\t$permute_sentences"
    echo -e "insert:\t\t\t$insert"
    echo -e "poisson-lambda:\t\t$poisson_lambda"
    echo -e "mask:\t\t\t$mask"
    echo ""
    echo -e "save_dir:\t\t$fairseq_save_dir"
    echo ""
    
    rm -rf "$fairseq_save_dir" && mkdir -p "$fairseq_save_dir"

    fairseq-train "$data_dir" \
        --save-dir "$fairseq_save_dir" \
        --arch "$model_config" \
        --wandb-project "bart-pretraining" \
        --seed "$seed" --fp16 \
        --lr 0.0004 `# adjust accordingly` \
        --optimizer adam \
        --lr-scheduler polynomial_decay `# linear (not in Fairseq) may improve budgeted training` \
        --weight-decay 0.01 \
        --criterion cross_entropy \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --clip-norm 0.1 \
        --share-all-embeddings `# defaults defined in bart-large architecture ` \
        --encoder-learned-pos `# defaults defined in bart-large architecture ` \
        --decoder-learned-pos `# defaults defined in bart-large architecture ` \
        --skip-invalid-size-inputs-valid-test \
        --log-format json \
        --dataset-impl mmap \
        --keep-interval-updates 1 \
        --keep-best-checkpoints 1 \
        --max-source-positions 256 `# restrict model input size for budgeted training` \
        --max-target-positions 256 `# restrict model output size for budgeted training` \
        --tokens-per-sample 256 `# size of each TokenBlockDataset (must be <= max-source-positions)` \
        --max-tokens 4096 `# batch-size (must be >= max-source-positions)` \
        --update-freq 1 `# effective batch size = max-tokens * update-freq ` \
        --max-update 250000 \
        --total-num-update 250000 `# requried by polynomial_decay scheduler - set == max-update (?) ` \
        --warmup-updates 2500 \
        --log-interval 1000 `# log every N updates` \
        --save-interval 1 `# save every N epochs also runs validation, so validation-interval == save-interval` \
        --save-interval-updates 5000 `# save every N updates also runs validation, so validation-interval-updates == save-interval-updates` \
        --num-workers 4 `# subprocesses to use for data loading` \
        --task "$task" `# bart's denoising task` \
        --mask-length span-poisson `# ["subword", "word", "span-poisson"]` \
        "$replace_length" `# 0 = no mask, 1 = 1 x <mask> for m tokens, -1 <mask> for each token` \
        "$rotate" `# document rotation: not used in final BART models` \
        "$mask_random" `# instead of using <mask>, use random token this often` \
        "$permute_sentences" `# sentence permutation: portion of sentences that are randomly shuffled in batch` \
        "$insert"`# insert this percentage of additional random tokens` \
        "$poisson_lambda" `# defined in paper as lambda=3` \
        "$mask" `# portion of words/subwords that will be masked` \
        | tee "$fairseq_save_dir/train.log"

    echo "$fairseq_save_dir"

    hf_save_dir="resources/models/seed_$seed/pt/hf_conv/${model_config}-${task}-${denoising_args}"
    convert_to_hf "$fairseq_save_dir" "$data_dir/../tok/tokenizer/" "$hf_save_dir"

}

train_dummy() {

    data_dir="resources/data/dummy_books1/bin"
    model_config="bart_small"
    save_dir="resources/models/pt/dummy_fairseq/$model_config"
    rm -rf $save_dir && mkdir -p $save_dir

    fairseq-train $data_dir \
        --save-dir $save_dir \
        --arch $model_config \
        --seed 4 --fp16 \
        --lr 0.0004 `# adjust accordingly` \
        --optimizer adam \
        --lr-scheduler polynomial_decay `# linear (not in Fairseq) may improve budgeted training` \
        --total-num-update 1000 `# requried by polynomial_decay scheduler` \
        --weight-decay 0.01 \
        --criterion cross_entropy \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --clip-norm 0.1 \
        --share-all-embeddings `# defaults defined in bart-large architecture ` \
        --encoder-learned-pos `# defaults defined in bart-large architecture ` \
        --decoder-learned-pos `# defaults defined in bart-large architecture ` \
        --skip-invalid-size-inputs-valid-test \
        --log-format json \
        --dataset-impl mmap \
        --keep-interval-updates 1 \
        --keep-best-checkpoints 1 \
        --max-source-positions 256 `# restrict model input size for budgeted training` \
        --max-target-positions 256 `# restrict model output size for budgeted training` \
        --tokens-per-sample 256 `# size of each TokenBlockDataset (must be <= max-source-positions)` \
        --max-tokens 4096 `# batch-size (must be >= max-source-positions)` \
        --update-freq 4 `# effective batch size = max-tokens * update-freq ` \
        --max-update 1000 \
        --warmup-updates 100 \
        --log-interval 100 `# log every N updates` \
        --save-interval 1 `# save every N epochs` \
        --save-interval-updates 100 `# save every N updates` \
        --keep-interval-updates 1 \
        --num-workers 0 `# subprocesses to use for data loading` \
        --task denoising `# bart's denoising task` \
        --mask-length span-poisson `# ["subword", "word", "span-poisson"]` \
        --replace-length 1 `# 0 = no mask, 1 = 1 x <mask> for m tokens, -1 <mask> for each token` \
        --rotate 0.0 `# document rotation: not used in final BART models` \
        --mask-random 0.0 `# instead of using <mask>, use random token this often` \
        --permute-sentences 1.0 `# sentence permutation: portion of sentences that are randomly shuffled in batch` \
        --insert 0.0 `# insert this percentage of additional random tokens` \
        --poisson-lambda 3.0 `# defined in paper as lambda=3` \
        --mask 0.0 `# portion of words/subwords that will be masked`

}

"$@"