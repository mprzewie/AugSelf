#!/bin/bash
#SBATCH --job-name=AugSelf_Barlow
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1

set -e

eval "$(conda shell.bash hook)"
conda activate uj

set -x

cd $HOME/uj/CASSLE

#export WANDB_API_KEY=8922102d08435f66d8640bbfa9caefd9c4e6be6d
#export WANDB_PROJECT=conditional_contrastive
#export WANDB_ENTITY=gmum
#export WANDB__SERVICE_WAIT=300

source lab_setup_datadirs.sh

FRAMEWORK=moco
BACKBONE=resnet18
PRETRAIN_DATASET=stl10
MAX_EPOCHS=200
FREQ=50
EVAL_FREQ=200
LR=0.05

BASE_DIR=$HOME/results/uj/CASSLE/


EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_augself"

OUT_DIR="${BASE_DIR}/${EXP_NAME}"

#nvidia-smi

##### pretraining

python pretrain_cond.py \
    --logdir $OUT_DIR \
    --framework $FRAMEWORK \
    --dataset $PRETRAIN_DATASET \
    --datadir ${datadirs["$PRETRAIN_DATASET"]} \
    --model $BACKBONE \
    --batch-size 256 \
    --max-epochs $MAX_EPOCHS \
    --ckpt-freq $FREQ --eval-freq $FREQ \
    --aug-inj-type proj-cat \
    --aug-cond grayscale \
    --aug-treatment mlp --aug-nn-depth 6 --aug-nn-width 64 #--aug-nn-out 256
#    --ss-crop 1 --ss-color 1 --ss-color-diff 1
    # aug conditionin


#bs 256
#### linear evaluation

source single_eval.sh

#### linear evaluation

#for DS in "cifar10" "cifar100" "food101" "mit67" "pets" "flowers" "caltech101" "cars" "aircraft" "dtd" "sun397";
#do
#  DS_DIR="${datadirs["$DS"]}"
#  DS_METRIC="${metrics["$DS"]}"
#
#
#  DS_OUT_FILE="linear_${DS}_${DS_METRIC}.txt"
#
#  CUDA_VISIBLE_DEVICES=0 python transfer_linear_eval.py \
#    --pretrain-data $PRETRAIN_DATASET \
#    --ckpt ${OUT_DIR}/ckpt-${MAX_EPOCHS}.pth \
#    --model $BACKBONE \
#    --dataset $DS \
#    --datadir $DS_DIR \
#    --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"
#done


#### few-shot evaluation

#for DS in "fc100" "cub200" "plant_disease";
#do
#  for N in 5;
#  do
#    for K in 1 5;
#    do
#      DS_OUT_FILE="few-shot_${DS}_${N}_${K}.txt"
#      DS_DIR="${datadirs["$DS"]}"
#
#      CUDA_VISIBLE_DEVICES=0 python transfer_few_shot.py \
#        --pretrain-data $PRETRAIN_DATASET \
#        --ckpt "${OUT_DIR}/ckpt-${MAX_EPOCHS}.pth" \
#        --model $BACKBONE \
#        --dataset $DS \
#        --datadir $DS_DIR \
#        --N 5 --K $K | tee "${OUT_DIR}/${DS_OUT_FILE}"
#    done
#  done
#
#done

#### linear (LOOC-like) evaluation

#for DS in "cub200" ; # "imagenet100"
#do
#  CUDA_VISIBLE_DEVICES=0 python transfer_looc_like.py \
#    -a resnet50 --lr 30.0 \
#    --batch-size 256 \
#    --dist-url 'tcp://localhost:10002' --world-size 1 --rank 0 --epochs 200  --schedule 120 160 \
#    --workers 4 \
#    --pretrained "${OUT_DIR}/ckpt-${MAX_EPOCHS}.pth" \
#    --dataset-name $DS ${datadirs["$DS"]}
#done


