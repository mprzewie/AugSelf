#!/bin/bash
#SBATCH --job-name=AugSelf_MOCO
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=dgxmatinf,dgxa100
#SBATCH --exclude=szerszen
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

FRAMEWORK=moco
BACKBONE=resnet50
PRETRAIN_DATASET="imagenet100"
MAX_EPOCHS=500
FREQ=50
EVAL_FREQ=50

BASE_DIR="/shared/results/przewiez/uj/AugSelf"

source setup_datadirs.sh



for AUG_TREATMENT in "mlp";
do
  for AUG_NN_DEPTH in 2 3;
  do
    for AUG_NN_WIDTH in 16;
    do
      for AUG_INJ in "proj-cat"
      do
        {
          EXTRA_ARGS="--aug-cond crop color --base-lr 0.05 --wd 1e-4 --ckpt-freq ${FREQ} --eval-freq ${FREQ} --num-workers 16 --distributed"
          SUFFIX="crop_color"
          source single_experiment.sh
        }
      done

    done
  done
done


