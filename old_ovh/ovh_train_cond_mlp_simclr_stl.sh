eval "$(conda shell.bash hook)"
conda activate AugSelfConditioning

set -x

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum

SRC_DIR="/home/mpyl/contrastif/AugSelf/"
RES_DIR="/storage/shared/results/mateusz.pyla/AugSelf/"

FRAMEWORK=simclr
BACKBONE=resnet18
PRETRAIN_DATASET=stl10
MAX_EPOCHS=200
FREQ=20
EVAL_FREQ=20
SEED=1997

EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_cond"
OUT_DIR="${RES_DIR}${EXP_NAME}"

CUDA="2"
#nvidia-smi

cd "${SRC_DIR}"

#### data
source "${SRC_DIR}ovh_setup_datadirs.sh"

for AUG_TREATMENT in "mlp";
do
  for AUG_NN_DEPTH in 2;
  do
    for AUG_NN_WIDTH in 8;
    do
      for CUR_LR in 0.05;
      do
        for AUG_INJ in "proj-cat"
        do
          {
            EXTRA_ARGS="--aug-cond crop color --base-lr ${CUR_LR} --wd 1e-4 --ckpt-freq ${FREQ} --eval-freq ${FREQ} --num-workers 16"
            SUFFIX="crop_color"
            source "${SRC_DIR}single_experiment.sh"
          }
        done
      done
    done
  done
done


