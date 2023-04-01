#!/bin/bash
#SBATCH --job-name=mp_marcin_cond_simsiam_4_32_005
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j_marcin_cond_simsiam_4_32_005.out

eval "$(conda shell.bash hook)"
conda activate AugSelfDgx

set -x

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum
#WANDB_DISABLE_CODE

SRC_DIR="/raid/NFS_SHARE/home/mateusz.pyla/contrastif/AugSelf/"
RES_DIR="/raid/NFS_SHARE/results/mateusz.pyla/contrastif/"

FRAMEWORK=simsiam
BACKBONE=resnet50
PRETRAIN_DATASET=imagenet100
IN100_PATH="/raid/NFS_SHARE/datasets/IN-100"
RESUME=-1
MAX_EPOCHS=500
FREQ=50
EVAL_FREQ=50
SEED=1997

EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_cond"
OUT_DIR="${RES_DIR}${EXP_NAME}"
export WANDB_DIR="${OUT_DIR}"

cd "${SRC_DIR}"

#### data
source "${SRC_DIR}ideas_setup_datadirs.sh"

for AUG_TREATMENT in "mlp";
do
  for AUG_NN_DEPTH in 8;
  do
    for AUG_NN_WIDTH in 16;
    do
      for CUR_LR in 0.03;
      do
        for AUG_INJ in "proj-cat"
        do
          {
            EXTRA_ARGS="--aug-cond crop color color_diff flip blur grayscale --base-lr ${CUR_LR} --wd 1e-4 --ckpt-freq ${FREQ} --eval-freq ${FREQ} --num-workers 16"
            SUFFIX="seed${SEED}"
            source "${SRC_DIR}ideas_single_experiment.sh"
          }
        done
      done
    done
  done
done
