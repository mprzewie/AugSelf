eval "$(conda shell.bash hook)"
conda activate AugSelfConditioning

set -x

cd /home/pyla/contrastive/AugSelf

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum

SRC_DIR="/home/pyla/contrastive/AugSelf/"
RES_DIR="/home/pyla/contrastive/AugSelfHelper/"

# source "${SRC_DIR}setup_datadirs.sh"


BACKBONE="resnet50"
MODEL_DIR="/home/pyla/contrastive/AugSelfHelper/"
CKPT_SEQ=500
MODEL_PATH="${MODEL_DIR}ckpt-${CKPT_SEQ}.pth"
pretrain="imagenet100"
IN100_PATH="/storage/shared/datasets/ImageNet100_ssl"
CIFAR100_PATH="/shared/sets/datasets/vision/" #cifar100

#### latent lin power evaluation

python transfer_feature_augmentation_interpolation.py \
    --model $BACKBONE --pretrain-data $pretrain --ckpt $MODEL_PATH --dataset "cifar100" --datadir $CIFAR100_PATH --augmentation colorblur


# for DS in "cifar10" "cifar100" "sun397" "caltech_101" "food101" "dtd" "pets" "cars" "aircraft" "flowers";
# do
#   {
#     DS_DIR="${datadirs["$DS"]}"
#     DS_METRIC="${metrics["$DS"]}"
#     DS_OUT_FILE="latent_linpower_${DS}_${DS_METRIC}.txt"

#     python transfer_feature_augmentation_interpolation.py \
#         --model $BACKBONE --pretrain-data $pretrain --ckpt $MODEL_PATH --dataset $DS --datadir ${datadirs["$DS"]} --augmentation colorblur
#   }
# done

# CUDA_VISIBLE_DEVICES=$CUDA python transfer_linear_eval.py \
#       --pretrain-data $PRETRAIN_DATASET \
#       --ckpt $CKPT_SEQ \
#       --model $BACKBONE \
#       --dataset $DS \
#       --datadir $DS_DIR \
#       --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"