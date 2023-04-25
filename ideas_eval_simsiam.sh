#!/bin/bash
#SBATCH --job-name=mp_marcin_augself_eval_simsiam
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j_marcin_augself_eval_simsiam.out

set -e

eval "$(conda shell.bash hook)"
conda activate AugSelfDgx

set -x

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum
#WANDB_DISABLE_CODE

SRC_DIR="/raid/NFS_SHARE/home/mateusz.pyla/contrastif/AugSelf/"
RES_DIR="/raid/NFS_SHARE/results/mateusz.pyla/contrastif/"
cd "${SRC_DIR}"

FRAMEWORK=simsiam
BACKBONE=resnet50
PRETRAIN_DATASET=imagenet100
IN100_PATH="/raid/NFS_SHARE/datasets/IN-100"

EXP_NAME="simsiam-resnet50-imagenet100_mlp_8_16_proj-cat_seed1997"
OUT_DIR="${RES_DIR}${EXP_NAME}"
export WANDB_DIR="${OUT_DIR}"

#CKPT_SEQ="${OUT_DIR}/ckpt-500.pth"
FREQ=50
EVAL_FREQ=50
CKPT_SEQ=$(for E in `seq $EVAL_FREQ $EVAL_FREQ $MAX_EPOCHS`; do echo "${OUT_DIR}/ckpt-${E}.pth"; done)

#### linear evaluation
source "${SRC_DIR}ideas_setup_datadirs.sh"
# "cifar10" "cifar100" "food101" "pets" "flowers" "caltech_101" "cars" "aircraft" "stl10" "sun397"
for DS in "caltech_101" "sun397" "stl10";
do
    DS_DIR="${datadirs["$DS"]}"
    DS_METRIC="${metrics["$DS"]}"
    DS_OUT_FILE="linear_${DS}_${DS_METRIC}.txt"
    python transfer_linear_eval.py \
      --pretrain-data $PRETRAIN_DATASET \
      --ckpt $CKPT_SEQ \
      --model $BACKBONE \
      --dataset $DS \
      --datadir $DS_DIR \
      --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"
done
