#! /bin/bash

# Training for ALl class
checkpoint="log/all/AE_AtlasNet_OurV2_SNall_tr2048_120epochs_npatches1_2019-03-08T12:48:46.689981/network.pth"
num_points=2500
nb_primitives="1"
env=SVR_AtlasNet_SNall_tr${num_points}_400epochs_npatches${nb_primitives}_pretrainedAE
python ./training/train_SVR_AtlasNet.py \
    --env ${env} \
    --model_preTrained_AE ${checkpoint} \
    --num_points ${num_points} \
    --nb_primitives ${nb_primitives} |& tee ${env}.txt

echo "Done"
exit


