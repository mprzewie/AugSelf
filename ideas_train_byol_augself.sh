#!/bin/bash
#SBATCH --job-name=mp_marcin_augself_003_byol
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j_marcin_augself_003_byol.out

eval "$(conda shell.bash hook)"
conda activate AugSelfDgx

set -x

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum
#WANDB_DISABLE_CODE

SRC_DIR="/raid/NFS_SHARE/home/mateusz.pyla/contrastif/AugSelf/"
RES_DIR="/raid/NFS_SHARE/results/mateusz.pyla/contrastif/"

FRAMEWORK=byol
BACKBONE=resnet50
PRETRAIN_DATASET=imagenet100
IN100_PATH="/raid/NFS_SHARE/datasets/IN-100"
RESUME=100
MAX_EPOCHS=500
FREQ=50
EVAL_FREQ=50
SEED=1997

EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_003_augself"
OUT_DIR="${RES_DIR}${EXP_NAME}"
export WANDB_DIR="${OUT_DIR}"

cd "${SRC_DIR}"

#### data
source "${SRC_DIR}ideas_setup_datadirs.sh"

python pretrain.py \
    --logdir $OUT_DIR \
    --framework $FRAMEWORK \
    --dataset $PRETRAIN_DATASET \
    --datadir $IN100_PATH \
    --batch-size 256 \
    --max-epochs $MAX_EPOCHS \
    --model $BACKBONE \
    --base-lr 0.03 --wd 1e-4 \
    --ckpt-freq $FREQ --eval-freq $EVAL_FREQ \
    --ss-crop 0.5 --ss-color 0.5 \
    --num-workers 16 \
    --resume $RESUME 
# --distributed
