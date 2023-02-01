#!/bin/bash
#SBATCH --job-name=AugSelf_MOCO
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=dgxmatinf,dgxa100
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
MAX_EPOCHS=200
FREQ=50
EVAL_FREQ=$MAX_EPOCHS


BASE_DIR="/shared/results/przewiez/uj/AugSelf"

source setup_datadirs.sh

EXTRA_ARGS="--aug-cond crop color"
SUFFIX="crop_color"

for AUG_TREATMENT in "mlp";
do
  for AUG_NN_DEPTH in 2 3;
  do
    for AUG_NN_WIDTH in 32 128 1024 2048;
    do
      for AUG_INJ in "proj-add" "proj-mul" "proj-cat"
      do
        source single_experiment.sh &
      done
      wait
    done
  done
done


