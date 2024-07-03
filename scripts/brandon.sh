#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {0..2}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
    --seed=${seeds[$i]} \
    --exp_name=0703_ppo_traffic=0.1 \
    --traffic_density=0.1 \
    --wandb \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done

# Loop over each GPU
for i in {3..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
    --seed=${seeds[$i]} \
    --exp_name=0703_ppo_traffic=0.2 \
    --traffic_density=0.2 \
    --wandb \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done
