#!/bin/bash
#SBATCH --job-name=AugSelf_MOCO
#SBATCH --gpus=2
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



for FRAMEWORK in "moco"; # "simsiam";
do
  for METHOD in "mlp_6_64_proj-cat_all_aug_w_color_diff_lr_0.03_seed_0"; #"augself" "baseline";
  do
    EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_${METHOD}"
    #
    OUT_DIR="${BASE_DIR}/${EXP_NAME}"
    source single_eval.sh


#for DS in "cifar10" "cifar100" "food101" "mit67" "pets" "flowers" "caltech101" "cars" "aircraft" "dtd" "sun397";
#    do
#      DS_DIR="${datadirs["$DS"]}"
#      DS_METRIC="${metrics["$DS"]}"
#
#
#      DS_OUT_FILE="linear_${DS}_${DS_METRIC}.txt"
#
#      CUDA_VISIBLE_DEVICES=0 python transfer_linear_eval.py \
#        --pretrain-data $PRETRAIN_DATASET \
#        --ckpt "${OUT_DIR}/ckpt-${MAX_EPOCHS}.pth" \
#        --model $BACKBONE \
#        --dataset $DS \
#        --datadir $DS_DIR \
#        --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"
#    done


#    for DS in "cub200"; # fc100" "cub200" "plant_disease" "imagenet100" "stl10"
#    do
#      for N in 5;
#      do
#        for K in 1 5;
#        do
#          DS_OUT_FILE="few-shot_${DS}_${N}_${K}.txt"
#          DS_DIR="${datadirs["$DS"]}"
#
#          CUDA_VISIBLE_DEVICES=0 python transfer_few_shot.py \
#            --pretrain-data $PRETRAIN_DATASET \
#            --ckpt "${OUT_DIR}/ckpt-${MAX_EPOCHS}.pth" \
#            --model $BACKBONE \
#            --dataset $DS \
#            --datadir $DS_DIR \
#            --N 5 --K $K | tee "${OUT_DIR}/${DS_OUT_FILE}"
#        done
#      done
#    done

#    for DS in "cub200" ; # "imagenet100"
#    do
#      CUDA_VISIBLE_DEVICES=0 python transfer_looc_like.py \
#        -a resnet50 --lr 30.0 \
#        --batch-size 256 \
#        --dist-url 'tcp://localhost:10002' --world-size 1 --rank 0 --epochs 200 --schedule 120 160  \
#        --workers 4 \
#        --pretrained "${OUT_DIR}/ckpt-${MAX_EPOCHS}.pth" \
#        --dataset-name $DS ${datadirs["$DS"]}
#    done

  done
done


