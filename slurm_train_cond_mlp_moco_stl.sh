#!/bin/bash
#SBATCH --job-name=AugSelf_MOCO
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=dgxmatinf,rtx2080
#SBATCH --exclude=szerszen
#SBATCH --cpus-per-task=8
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
BACKBONE=resnet18
PRETRAIN_DATASET=stl10


BASE_DIR="/shared/results/przewiez/uj/AugSelf"

for AUG_TREATMENT in "mlp";
do
  for AUG_NN_DEPTH in 2 3 1;
  do
    for AUG_NN_WIDTH in 16 32 64;
    do
      EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_${AUG_TREATMENT}_${AUG_NN_DEPTH}_${AUG_NN_WIDTH}"
      OUT_DIR="${BASE_DIR}/${EXP_NAME}"

      #nvidia-smi

      ##### pretraining

      CUDA_VISIBLE_DEVICES=0 python pretrain_cond.py \
          --logdir $OUT_DIR \
          --framework moco \
          --dataset stl10 \
          --datadir /shared/sets/datasets/vision/stl10/ \
          --model resnet18 \
          --batch-size 256 \
          --max-epochs 200 \
          --aug-treatment $AUG_TREATMENT \
          --aug-nn-depth $AUG_NN_DEPTH --aug-nn-width $AUG_NN_WIDTH

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
          --pretrain-data stl10 \
          --ckpt ${OUT_DIR}/ckpt-200.pth \
          --model $BACKBONE \
          --dataset $DS \
          --datadir $DS_DIR \
          --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"
      done
    done
  done
done


