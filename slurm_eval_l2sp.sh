#!/bin/bash
#SBATCH --job-name=AugSelf_MOCO
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1

set -e

eval "$(conda shell.bash hook)"
conda activate uj

set -x

cd $HOME/uj/AugSelf

export WANDB_API_KEY=8922102d08435f66d8640bbfa9caefd9c4e6be6d
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum
export WANDB__SERVICE_WAIT=300


FRAMEWORK=moco
BACKBONE=resnet50
PRETRAIN_DATASET=imagenet100
MAX_EPOCHS=500


BASE_DIR="/raid/NFS_SHARE/results/marcin.przewiezlikowski/uj/AugSelf/"


#EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_augself"
#
#OUT_DIR="${BASE_DIR}/${EXP_NAME}"

#nvidia-smi

##### pretraining

#python pretrain.py \
#    --logdir $OUT_DIR \
#    --framework $FRAMEWORK \
#    --dataset $PRETRAIN_DATASET \
#    --datadir /shared/sets/datasets/vision/IN-100/ \
#    --model $BACKBONE \
#    --batch-size 256 \
#    --max-epochs $MAX_EPOCHS \
#    --base-lr 0.05 --wd 1e-4 \
#    --ckpt-freq 50 --eval-freq 50 \
#    --ss-crop 0.5 --ss-color 0.5 \
#    --num-workers 16 --distributed

#### linear evaluation

source setup_datadirs.sh



for ALPHA in 1 0.1 0.01 0.001; do
    for LR in 0.01 0.02  0.005; do
#        for FRAMEWORK in "moco"; # "simsiam";
#        do
#          for METHOD in "mlp_6_64_proj-cat_all_aug_w_color_diff_lr_0.03_seed_0" "augself_OFFICIAL" "baseline_OFFICIAL";
#          do
#            for DS in "caltech101" "cifar10" "pets" "food101" "cub200" ; do # "cub200"
#                DS_DIR="${datadirs["$DS"]}"
#                EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_${METHOD}"
#                OUT_DIR="${BASE_DIR}/${EXP_NAME}"
#                python transfer_l2sp.py \
#                    --ckpt "${OUT_DIR}/ckpt-${EVAL_EPOCHS}.pth" \
#                    --pretrain-data $PRETRAIN_DATASET \
#                    --dataset $DS \
#                    --datadir $DS_DIR \
#                    --model $BACKBONE \
#                    --l2sp-lr-init $LR --l2sp-alpha $ALPHA
#            done
#          done
#        done

        for FRAMEWORK in "simsiam";
        do
          for METHOD in "mlp_6_64_proj-cat_bn_proj_our_lr_0.05_seed_0" "augself_OFFICIAL" "baseline_OFFICIAL";
          do
            for DS in "cifar10" "pets" "mit67" "food101" "caltech101"; do # "cub200"
                DS_DIR="${datadirs["$DS"]}"
                EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_${METHOD}"
                OUT_DIR="${BASE_DIR}/${EXP_NAME}"
                python transfer_l2sp.py \
                    --ckpt "${OUT_DIR}/ckpt-${EVAL_EPOCHS}.pth" \
                    --pretrain-data $PRETRAIN_DATASET \
                    --dataset $DS \
                    --datadir $DS_DIR \
                    --model $BACKBONE \
                    --l2sp-lr-init $LR --l2sp-alpha $ALPHA
            done
          done
        done
    done
done

