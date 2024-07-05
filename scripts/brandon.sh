#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


# Loop over each GPU
for i in {4..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
    --seed=${seeds[$i]} \
    --exp_name="0705_ppo_traffic=0.1_lane_line_detector=120" \
    --traffic_density=0.1 \
    --wandb \
    --lane_line_detector=120 \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done


# Loop over each GPU
for i in {6..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
    --seed=${seeds[$i]} \
    --exp_name="0705_ppo_traffic=0.1_lane_line_detector=120_vehicle_detector=240" \
    --traffic_density=0.1 \
    --wandb \
    --lane_line_detector=120 \
    --vehicle_detector=240 \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done
