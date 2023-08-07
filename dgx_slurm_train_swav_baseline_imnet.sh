#!/bin/bash
#SBATCH --job-name=AugSelf_SWAV
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=20
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

source dgx_setup_datadirs.sh

FRAMEWORK=swav
BACKBONE=resnet50
PRETRAIN_DATASET=imagenet100
MAX_EPOCHS=500
FREQ=50
EVAL_FREQ=500
LR=0.05

BASE_DIR="/raid/NFS_SHARE/results/marcin.przewiezlikowski/uj/AugSelf/"


EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_baseline_san_check_dgx"
#EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_augself_san_check_dgx"

OUT_DIR="${BASE_DIR}/${EXP_NAME}"


 python pretrain.py \
     --logdir $OUT_DIR \
     --framework swav \
     --dataset imagenet100 \
     --datadir ${datadirs["$PRETRAIN_DATASET"]} \
     --batch-size 256 \
     --max-epochs 500 \
     --model resnet50 \
     --base-lr 0.05 --wd 1e-4 \
     --ckpt-freq 50 --eval-freq 50 \
     --num-workers 16 --distributed #--ss-crop 0.5 --ss-color 0.5
nvidia-smi


source single_eval_ablation.sh
