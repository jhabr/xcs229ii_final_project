#!/bin/bash
CUDA_VISIBLE_DEVICES=0
python3 "$(pwd)/transformers/trans_u_net/train.py" --dataset ISIC --dataset_size 10 --vit_name R50-ViT-B_16
