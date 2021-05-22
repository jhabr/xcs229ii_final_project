#!/bin/bash
CUDA_VISIBLE_DEVICES=0

if [ -z ${PYTHONPATH+x} ]
then
  echo "PYTHONPATH not set. Exiting..."; exit 1
  else echo "PYTHONPATH set to $PYTHONPATH."
fi

(python3 "$(pwd)/transformers/trans_u_net/train.py" --dataset ISIC --dataset_size 10 --vit_name R50-ViT-B_16 2>&1) >> "$PYTHONPATH"/logs/transunet.log &
