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
penalty=(0.5 1 2 5)
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=0 \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal.py \
    --wandb \
    --seed=0 \
    --exp_name=td3_multigoal_pen${penalty[$i]}-v4 \
    --penalty=${penalty[$i]} \
    > "brandon-td3-${penalty[$i]}.log" 2>&1 &
done


# Let's say we want to grid search penalty = 0.5, 1, 2, 5, 10
# Loop over it:
penalty=(0.5 1 2 5)
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=1 \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_sac.py \
    --wandb \
    --seed=0 \
    --exp_name=sac_multigoal_pen${penalty[$i]}-v4 \
    --penalty=${penalty[$i]} \
    > "brandon-sac-${penalty[$i]}.log" 2>&1 &
done
