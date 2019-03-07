#! /bin/bash

num_points=2048
for nb_primitives in "1" "2" "25"; do
    env=AE_AtlasNet_OurV2_SNall_tr2048_120epochs_npatches${nb_primitives}
    python ./training/train_AE_AtlasNet_our.py \
        --env ${env} \
        --class_choice all \
        --num_points ${num_points} \
        --nb_primitives ${nb_primitives}
        --super_points ${num_points} |& tee ${env}.txt
done


# env=AE_AtlasNet_OurV2_SNall_tr2048_1000epochs
# num_points=2048
# python ./training/train_AE_AtlasNet_our.py \
#     --env ${env} \
#     --class_choice all \
#     --num_points ${num_points} \
#     --nepoch 1000 \
#     --super_points ${num_points} |& tee ${env}.txt


