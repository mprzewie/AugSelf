#!/bin/bash
#SBATCH --job-name=AugSelf_MOCO
#SBATCH --qos=big
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --partition=dgxmatinf,dgxa100
#SBATCH --exclude=szerszen
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1

set -e

eval "$(conda shell.bash hook)"
conda activate uj

set -x

cd $HOME/uj/AugSelf

export WANDB_API_KEY=8922102d08435f66d8640bbfa9caefd9c4e6be6d
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum

FRAMEWORK=moco
BACKBONE=resnet50
PRETRAIN_DATASET=imagenet100
MAX_EPOCHS=500


BASE_DIR="/shared/results/przewiez/uj/AugSelf"


EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_augself"

OUT_DIR="${BASE_DIR}/${EXP_NAME}"

#nvidia-smi

##### pretraining

python pretrain.py \
    --logdir $OUT_DIR \
    --framework $FRAMEWORK \
    --dataset $PRETRAIN_DATASET \
    --datadir /shared/sets/datasets/vision/IN-100/ \
    --model $BACKBONE \
    --batch-size 256 \
    --max-epochs $MAX_EPOCHS \
    --base-lr 0.05 --wd 1e-4 \
    --ckpt-freq 50 --eval-freq 50 \
    --ss-crop 0.5 --ss-color 0.5 \
    --num-workers 16 --distributed

#### linear evaluation

declare -A datadirs
for DS in "cifar10" "cifar100" "pets" "flowers" "caltech101" "cars" "aircraft" "sun397";
do
  datadirs[$DS]="/shared/sets/datasets/vision"
done

datadirs["food101"]="/shared/sets/datasets/vision/food_101/"
datadirs["mit67"]="/shared/sets/datasets/vision/mit67_indoor_scenes/indoorCVPR_09/images_train_test/"
datadirs["stl10"]="/shared/sets/datasets/vision/stl10/"
datadirs["dtd"]="/home/przewiez/Downloads/dtd/"

declare -A metrics
for DS in ${!datadirs[@]};
do
  metrics[$DS]="top1"
done

for DS in "pets" "flowers" "caltech101" "aircraft"; do
  metrics[$DS]="class-avg"
done


for DS in ${!datadirs[@]};
do
  DS_DIR="${datadirs["$DS"]}"
  DS_METRIC="${metrics["$DS"]}"


  DS_OUT_FILE="linear_${DS}_${DS_METRIC}.txt"

  CUDA_VISIBLE_DEVICES=0 python transfer_linear_eval.py \
    --pretrain-data $PRETRAIN_DATASET \
    --ckpt "${OUT_DIR}/ckpt-${MAX_EPOCHS}.pth" \
    --model $BACKBONE \
    --dataset $DS \
    --datadir $DS_DIR \
    --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"
done



