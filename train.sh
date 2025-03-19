#!/usr/bin/env bash
set -ex


# sft
accelerate launch \
    --config_file configs/ds_z3.yaml \
    train_sft.py \
    --config configs/recipes/qwen2.5_7b_sft.yaml

# qa-sft+ste
accelerate launch \
    --config_file configs/ds_z3.yaml \
    train_qa_sft_ste.py \
    --config configs/recipes/qwen2.5_7b_qa_sft_ste.yaml

# qa-sft+roste
accelerate launch \
    --config_file configs/ds_z3.yaml \
    train_qa_sft_roste.py \
    --config configs/recipes/qwen2.5_7b_qa_sft_roste.yaml