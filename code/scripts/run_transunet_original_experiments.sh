#!/bin/bash
CUDA_VISIBLE_DEVICES=0

if [ -z ${PYTHONPATH+x} ]
then
  echo "PYTHONPATH not set. Exiting..."; exit 1
  else echo "PYTHONPATH set to $PYTHONPATH."
fi

train_script="$(pwd)/transformers/trans_u_net/train.py"

#experiment 00
train_args=(--dataset ISIC --dataset_size 10 --batch_size=2 --max_epochs=10 --vit_name R50-ViT-B_16)
(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &

#experiment 50
#train_args=(--dataset ISIC --vit_name R50-ViT-B_16)
#(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &
