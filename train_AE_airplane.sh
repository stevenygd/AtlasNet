#! /bin/bash

num_points=2048
for n_primitives in "1" "2"; do
    env=AE_AtlasNet_OurV2_airplane_tr2048_1000epochs_patches${n_primitives}
    python ./training/train_AE_AtlasNet_our.py \
        --env ${env} \
        --class_choice airplane \
        --num_points ${num_points} \
        --nepoch 1000 \
        --nb_primitives ${n_primitives} \
        --super_points ${num_points} |& tee ${env}.txt

    env=AE_AtlasNet_OurV2_airplane_patches${n_primitives}
    python ./training/train_AE_AtlasNet.py \
        --env ${env} \
        --class_choice airplane \
        --num_points ${num_points} \
        --nb_primitives ${n_primitives} \
        --super_points ${num_points} |& tee ${env}.txt

done
