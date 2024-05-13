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
--exp_name=td3_multigoal_pen2_rew1 \
--penalty=2 \
--driving_reward=1.0 \
> "brandon-35.log" 2>&1 &


CUDA_VISIBLE_DEVICES=0 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=0 \
--exp_name=td3_multigoal_pen2_rew2 \
--penalty=2 \
--driving_reward=2.0 \
> "brandon-83.log" 2>&1 &


CUDA_VISIBLE_DEVICES=0 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=0 \
--exp_name=td3_multigoal_pen2_rew5 \
--penalty=2 \
--driving_reward=5.0 \
> "brandon-31.log" 2>&1 &


CUDA_VISIBLE_DEVICES=0 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=0 \
--exp_name=td3_multigoal_pen2_rew10 \
--penalty=2 \
--driving_reward=10.0 \
> "brandon-30.log" 2>&1 &
