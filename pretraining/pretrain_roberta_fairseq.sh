#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# this training command is adapted from the original provided by M. Lewis
# https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320

DATA_DIR="resources/data/books1/bin"
MODEL_CONFIG="roberta_small"
TASK="masked_lm"
SAVE_DIR=""

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
        -*|--*)
            echo "Unknown option $1" && exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done

if [ -z "$SEED" ]; then
    echo "ERROR: seed not provided" && exit 1
fi

if TASK="masked_lm"; then
    MODEL_ID="$MODEL_CONFIG-MLM"
else
    echo "ERROR: task ($TASK) not supported" && exit 1
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="resources/models/seed_$SEED/pt/fairseq/$MODEL_ID"
fi

mkdir -p "$SAVE_DIR"

echo ""
echo "Launching model training with the following args:"
echo ""
echo -e "$DATA_DIR"
echo -e "arch:\t\t$MODEL_CONFIG"
echo -e "task:\t\t\t$TASK"
echo -e "seed:\t\t\t$SEED"
echo -e "save-dir:\t\t$SAVE_DIR"
echo ""

fairseq-train "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --arch "$MODEL_CONFIG" \
    --wandb-project "$MODEL_CONFIG-pretraining" \
    --seed "$SEED" --fp16 \
    --lr 0.0005 `# adjust accordingly` \
    --optimizer adam \
    --lr-scheduler polynomial_decay `# linear (not in Fairseq) may improve budgeted training` \
    --weight-decay 0.01 \
    --criterion "masked_lm" \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --clip-norm 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --log-format json \
    --dataset-impl mmap \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
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
    | tee "$SAVE_DIR/train.log"
