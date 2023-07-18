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
SEED=0

BASE_DIR="/raid/NFS_SHARE/results/marcin.przewiezlikowski/uj/AugSelf/"


#EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_baseline"

OUT_DIR="${BASE_DIR}/${EXP_NAME}"

#nvidia-smi

##### pretraining


for AUG_TREATMENT in "mlp";
do
  for AUG_NN_DEPTH in 6;
  do
    for AUG_NN_WIDTH in 64;
    do
      for AUG_INJ in "proj-cat"
#      for AUG_INJ in "proj-none"
      do
        {
          EXTRA_ARGS="--aug-cond crop color color_diff flip blur grayscale --base-lr ${LR} --wd 1e-4 --ckpt-freq ${FREQ} --eval-freq ${FREQ} --num-workers 16 --seed ${SEED} --distributed"
          SUFFIX="ideas_all_aug_w_color_diff_lr_${LR}_seed_${SEED}_500ep"

#          EXTRA_ARGS="--ss-crop 0.5 --ss-color 0.5 --base-lr ${LR} --wd 1e-4 --ckpt-freq ${FREQ} --eval-freq ${FREQ} --num-workers 16 --seed ${SEED} --distributed"
#          SUFFIX="augself"
#
#          EXTRA_ARGS="--base-lr ${LR} --wd 1e-4 --ckpt-freq ${FREQ} --eval-freq ${FREQ} --num-workers 16 --seed ${SEED} --distributed"
#          SUFFIX="baseline"

          source single_experiment.sh
        }
      done
    done
  done
done

