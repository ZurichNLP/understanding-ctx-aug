#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# this training command is adapted from the original provided by M. Lewis
# https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320

# default values
HEADS_PROB=0.5

# argument parser
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2" # "resources/data/books1/bin"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --replace_length)
            REPLACE_LENGTH="$2"
            shift 2 # past argument
            ;;
        --mask_random)
            MASK_RANDOM="$2"
            shift 2 # past argument
            ;;
        --rotate)
            ROTATE="$2"
            shift 2
            ;;
        --permute_sentences)
            PERMUTE_SENTENCES="$2"
            shift 2 # past argument
            ;;
        --insert)
            INSERT="$2"
            shift 2 # past argument
            ;;
        --poisson_lambda)
            POISSON_LAMBDA="$2"
            shift 2 # past argument
            ;;
        --mask)
            MASK="$2"
            shift 2 # past argument
            ;; 
        --denoising_method)
            DENOISING_METHOD="$2"
            shift 2 # past argument
            ;;
        --heads_prob)
            HEADS_PROB="$2"
            shift 2 # past argument
            ;;
        -*|--*)
            echo "Unknown option $1" && exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done

echo ""
echo "Launching model training with the following args:"
echo ""
echo -e "$DATA_DIR"
echo -e "arch:\t\t$MODEL_CONFIG"
echo -e "task:\t\t\t$TASK"
echo -e "denoising_method:\t$DENOISING_METHOD"
echo -e "heads_prob:\t\t$HEADS_PROB"
echo -e "seed:\t\t\t$SEED"
echo -e "replace-length:\t\t$REPLACE_LENGTH"
echo -e "mask-random:\t\t$MASK_RANDOM"
echo -e "rotate:\t\t\t$ROTATE"
echo -e "permute-sentences:\t$PERMUTE_SENTENCES"
echo -e "insert:\t\t\t$INSERT"
echo -e "poisson-lambda:\t\t$POISSON_LAMBDA"
echo -e "mask:\t\t\t$MASK"
echo -e "save-dir:\t\t$SAVE_DIR"
echo ""

fairseq-train "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --arch "$MODEL_CONFIG" \
    --wandb-project "bart-pretraining" \
    --seed "$SEED" --fp16 \
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
    --task "$TASK" `# bart's denoising task` \
    --mask-length span-poisson `# ["subword", "word", "span-poisson"]` \
    --replace-length "$REPLACE_LENGTH" `# 0 = no mask, 1 = 1 x <mask> for m tokens, -1 <mask> for each token` \
    --rotate "$ROTATE" `# document rotation: not used in final BART models` \
    --mask-random "$MASK_RANDOM" `# instead of using <mask>, use random token this often` \
    --permute-sentences "$PERMUTE_SENTENCES" `# sentence permutation: portion of sentences that are randomly shuffled in batch` \
    --insert "$INSERT"`# insert this percentage of additional random tokens` \
    --poisson-lambda "$POISSON_LAMBDA" `# defined in paper as lambda=3` \
    --mask "$MASK" `# portion of words/subwords that will be masked` \
    --denoising-method "$DENOISING_METHOD" `# ["default", "bart", "t5", "mass"]` \
    --heads-prob "$HEADS_PROB" `# probability of using BART or T5 style denosing (only required for BART5 denoising method)`

echo "$SAVE_DIR"