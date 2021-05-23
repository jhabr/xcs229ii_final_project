#!/bin/bash
CUDA_VISIBLE_DEVICES=0

if [ -z ${PYTHONPATH+x} ]
then
  echo "PYTHONPATH not set. Exiting..."; exit 1
  else echo "PYTHONPATH set to $PYTHONPATH."
fi

train_script="$(pwd)/transformers/trans_u_net/train.py"

#experiment 00
train_args=(--dataset ISIC --train_dataset_size 10 --valid_dataset_size 10 --batch_size=2 --max_epochs=2 --vit_name R50-ViT-B_16)
(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &

#experiment 50
#train_args=(--dataset ISIC --vit_name R50-ViT-B_16)
#(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &

#experiment 51
#train_args=(--dataset ISIC --vit_name R50-ViT-B_16 --img_size 128)
#(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &

#experiment 53
#train_args=(--dataset ISIC --vit_name R50-ViT-B_32 --img_size 128)
#(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &

#experiment 54
#train_args=(--dataset ISIC --vit_name ViT-B_16 --img_size 128)
#(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &

#experiment 55
#train_args=(--dataset ISIC --vit_name ViT-B_16 --img_size 128)
#(python3 "$train_script" "${train_args[@]}" 2>&1) >> "$PYTHONPATH"/logs/transunet_original.log &