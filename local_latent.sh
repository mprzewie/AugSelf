eval "$(conda shell.bash hook)"
conda activate AugSelfConditioning

set -x

cd /home/mpyl/contrastif/AugSelf 

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum

SRC_DIR="/home/mateuszpyla/Pulpit/phd/constrastif/AugSelf/"
RES_DIR="/home/mateuszpyla/fromovh/mateusz.pyla/AugSelf/"

source "${SRC_DIR}ovh_setup_datadirs.sh"


BACKBONE="resnet50"
MODEL_DIR="/home/mateuszpyla/fromovh/mateusz.pyla/AugSelf/simsiam-resnet50-imagenet100_mlp_2_16_proj-cat_crop_color/"
CKPT_SEQ=500
MODEL_PATH="${MODEL_DIR}ckpt-${CKPT_SEQ}.pth"
pretrain="imagenet100"
IN100_PATH="/storage/shared/datasets/ImageNet100_ssl"
CIFAR100_PATH="/home/mateuszpyla/Pulpit/phd/data/" #cifar100

#### latent lin power evaluation

python transfer_feature_augmentation_interpolation.py \
    --model $BACKBONE --pretrain-data $pretrain --ckpt $MODEL_PATH --dataset "cifar100" --datadir $CIFAR100_PATH


# for DS in "cifar10" "cifar100" "sun397" "caltech_101" "food101" "dtd" "pets" "cars" "aircraft" "flowers";
#do
#  {
#    DS_DIR="${datadirs["$DS"]}"
#    DS_METRIC="${metrics["$DS"]}"
#    DS_OUT_FILE="latent_linpower_${DS}_${DS_METRIC}.txt"

#    CUDA_VISIBLE_DEVICES=$CUDA python transfer_feature_invariance.py \
#    --model $BACKBONE --pretrain-data $pretrain --ckpt $MODEL_PATH --dataset $DS --datadir ${datadirs["$DS"]}
    
#  }
#done

# CUDA_VISIBLE_DEVICES=$CUDA python transfer_linear_eval.py \
#       --pretrain-data $PRETRAIN_DATASET \
#       --ckpt $CKPT_SEQ \
#       --model $BACKBONE \
#       --dataset $DS \
#       --datadir $DS_DIR \
#       --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"