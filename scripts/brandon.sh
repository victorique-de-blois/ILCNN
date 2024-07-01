#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_td3.py \
    --seed=${seeds[$i]} \
    --exp_name=0701_td3_general \
    --wandb \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done
