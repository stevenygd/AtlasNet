#! /bin/bash

env=AE_AtlasNet_OurV2_SNall_tr2048
num_points=2048
python ./training/train_AE_AtlasNet_our.py \
    --env ${env} \
    --class_choice all \
    --num_points ${num_points} \
    --super_points ${num_points} |& tee ${env}.txt


env=AE_AtlasNet_OurV2_SNall_tr2048_1000epochs
num_points=2048
python ./training/train_AE_AtlasNet_our.py \
    --env ${env} \
    --class_choice all \
    --num_points ${num_points} \
    --nepoch 1000 \
    --super_points ${num_points} |& tee ${env}.txt


