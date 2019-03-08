#! /bin/bash

num_points=2500
for nb_primitives in "1" "2"; do
# for nb_primitives in "25"; do
    env=SVR_AtlasNet_SNall_tr${num_points}_400epochs_npatches${nb_primitives}
    python ./training/train_SVR_AtlasNet.py \
        --env ${env} \
        --num_points ${num_points} \
        --nb_primitives ${nb_primitives} |& tee ${env}.txt
done
echo "Done"
exit


# env=AE_AtlasNet_OurV2_SNall_tr2048_1000epochs
# num_points=2048
# python ./training/train_AE_AtlasNet_our.py \
#     --env ${env} \
#     --class_choice all \
#     --num_points ${num_points} \
#     --nepoch 1000 \
#     --super_points ${num_points} |& tee ${env}.txt


