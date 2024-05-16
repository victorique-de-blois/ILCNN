#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
#for i in {0..3}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_sac.py \
#    --seed=${seeds[$i]} \
#    > "brandon-seed${seeds[$i]}.log" 2>&1 &
#done

# Let's say we want to grid search penalty = 0.5, 1, 2, 5, 10
# Loop over it:
CUDA_VISIBLE_DEVICES=7 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
--wandb \
--seed=100 \
--lr=1e-4 \
--exp_name=td3_multigoal_0516 \
> "brandon-td3.log" 2>&1 &

CUDA_VISIBLE_DEVICES=7 \
nohup python pvp/experiments/metadrive/train_metadrive_multigoal_sac.py \
--wandb \
--seed=100 \
--lr=1e-4 \
--exp_name=sac_multigoal_0516 \
> "brandon-sac.log" 2>&1 &
