#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
#for i in {0..3}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
#    --seed=${seeds[$i]} \
#    > "brandon-seed${seeds[$i]}.log" 2>&1 &
#done

CUDA_VISIBLE_DEVICES=0 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=0 \
--exp_name=td3_multigoal_pen5 \
--penalty=5 \
> "brandon-111.log" 2>&1 &


CUDA_VISIBLE_DEVICES=1 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=0 \
--exp_name=td3_multigoal_pen10 \
--penalty=10 \
> "brandon-2221.log" 2>&1 &


CUDA_VISIBLE_DEVICES=2 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=0 \
--exp_name=td3_multigoal_pen2 \
--penalty=2 \
> "brandon-333.log" 2>&1 &


CUDA_VISIBLE_DEVICES=3 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=0 \
--exp_name=td3_multigoal_pen \
--penalty=0 \
> "brandon-333.log" 2>&1 &

