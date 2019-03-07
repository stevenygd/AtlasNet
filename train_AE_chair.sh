#! /bin/bash


num_points=2048
for n_patches in "1" "2"; do
    env=AE_AtlasNet_OurV2_chair_tr2048_npatch${n_patches}
    python ./training/train_AE_AtlasNet_our.py \
        --env ${env} \
        --class_choice chair \
        --num_points ${num_points} \
        --nb_primitives ${n_patches} \
        --super_points ${num_points} |& tee ${env}.txt

    env=AE_AtlasNet_OurV2_chair_tr2048_1000epochs_npatch${n_patches}
    python ./training/train_AE_AtlasNet_our.py \
        --env ${env} \
        --class_choice chair \
        --num_points ${num_points} \
        --nb_primitives ${n_patches} \
        --nepoch 1000 \
        --super_points ${num_points} |& tee ${env}.txt

done
