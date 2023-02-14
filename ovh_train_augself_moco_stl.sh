eval "$(conda shell.bash hook)"
conda activate AugSelfConditioning

set -x

cd /home/mpyl/contrastif/AugSelf 

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum

SRC_DIR="/home/mpyl/contrastif/AugSelf/"
RES_DIR="/storage/shared/results/mateusz.pyla/AugSelf/"

FRAMEWORK=moco
BACKBONE=resnet18
PRETRAIN_DATASET=stl10
STL10_PATH="/storage/shared/datasets/stl10"
MAX_EPOCHS=200
FREQ=20
EVAL_FREQ=20
SEED=1997

EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_augself"
OUT_DIR="${RES_DIR}${EXP_NAME}"

CUDA="0,1"
#nvidia-smi

#### data
echo "${SRC_DIR}ovh_setup_datadirs.sh"
source "${SRC_DIR}ovh_setup_datadirs.sh"

#### pretraining

CUDA_VISIBLE_DEVICES=$CUDA python pretrain.py \
   --logdir $OUT_DIR \
   --framework $FRAMEWORK \
   --dataset $PRETRAIN_DATASET \
   --datadir $STL10_PATH \
   --model $BACKBONE \
   --batch-size 256 \
   --max-epochs $MAX_EPOCHS \
   --base-lr 0.05 --wd 1e-4 \
   --ckpt-freq $FREQ --eval-freq $EVAL_FREQ \
   --ss-crop 0.5 --ss-color 0.5 \
   --num-workers 16 \
   --distributed
   
#--seed $SEED

#### linear evaluation

source "${SRC_DIR}single_eval.sh"
