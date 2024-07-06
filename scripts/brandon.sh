#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)

#for i in {0..1}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_ppo_metadrive.py \
#    --seed=${seeds[$i]} \
#    --exp_name="0706_ppo_regression" \
#    --wandb \
#    --wandb_projec="brandon" \
#    --wandb_team="drivingforce" \
#    > "brandon-seed${seeds[$i]}.log" 2>&1 &
#done


for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
    --seed=${seeds[$i]} \
    --exp_name="0706_ppo_traffic=0.1_lane_line_detector=0_NATIVE_REWARD" \
    --traffic_density=0.1 \
    --wandb \
    --lane_line_detector=0 \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done


for i in {2..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
    --seed=${seeds[$i]} \
    --exp_name="0706_ppo_traffic=0.0_lane_line_detector=0_NATIVE_REWARD" \
    --traffic_density=0.0 \
    --wandb \
    --lane_line_detector=0 \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {4..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
    --seed=${seeds[$i]} \
    --exp_name="0706_ppo_traffic=0.1_lane_line_detector=0_NATIVE_REWARD" \
    --traffic_density=0.1 \
    --wandb \
    --lane_line_detector=0 \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done

