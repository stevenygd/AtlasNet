#! /bin/bash

env=AE_AtlasNet_OurV2_airplane_tr2048_1000epochs
num_points=2048
python ./training/train_AE_AtlasNet_our.py \
    --env ${env} \
    --class_choice airplane \
    --num_points ${num_points} \
    --nepoch 1000 \
    --super_points ${num_points} |& tee ${env}.txt


# env=AE_AtlasNet_OurV2_airplane
# num_points=2048
# python ./training/train_AE_AtlasNet.py \
#     --env ${env} \
#     --class_choice airplane \
#     --num_points ${num_points} \
#     --super_points ${num_points} |& tee ${env}.txt
