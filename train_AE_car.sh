#! /bin/bash

num_points=2048
for n_patches in "1" "2"; do

    env=AE_AtlasNet_OurV2_car_tr2048_npatches${n_patches}
    python ./training/train_AE_AtlasNet_our.py \
        --env ${env} \
        --class_choice car \
        --num_points ${num_points} \
        --nb_primitives ${n_patches} \
        --super_points ${num_points} |& tee ${env}.txt

    env=AE_AtlasNet_OurV2_car_tr2048_1000epochs_npatches${n_patches}
    python ./training/train_AE_AtlasNet_our.py \
        --env ${env} \
        --class_choice car \
        --num_points ${num_points} \
        --nepoch 1000 \
        --nb_primitives ${n_patches} \
        --super_points ${num_points} |& tee ${env}.txt


done
