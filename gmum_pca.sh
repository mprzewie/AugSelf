eval "$(conda shell.bash hook)"
conda activate AugSelfConditioning

set -x

cd /home/pyla/contrastive/AugSelf

export WANDB_API_KEY=fbe8977ced9962ba4c826b6e012e35dad2c3f044
export WANDB_PROJECT=conditional_contrastive
export WANDB_ENTITY=gmum

SRC_DIR="/home/pyla/contrastive/AugSelf/"
RES_DIR="/home/pyla/contrastive/AugSelfHelper/"


BACKBONE="resnet50"
MODEL_DIR="/home/pyla/contrastive/AugSelfHelper/"
CKPT_SEQ=500
MODEL_PATH="${MODEL_DIR}ckpt-${CKPT_SEQ}.pth"
pretrain="imagenet100"
IN100_PATH="/storage/shared/datasets/ImageNet100_ssl"
CIFAR100_PATH="/shared/sets/datasets/vision/"

#### latent pca

PROBAB=True

FIRST="/shared/results/przewiez/uj/AugSelf/"
LAST="ckpt-500.pth"
PATH1="moco-resnet50-imagenet100_augself_OFFICIAL/"
PATH2="moco-resnet50-imagenet100_augself_lr_0.03"
PATH3="moco-resnet50-imagenet100_baseline_OFFICIAL"
PATH4="moco-resnet50-imagenet100_mlp_6_64_proj-cat_all_aug_lr_0.03"
PATH5="moco-resnet50-imagenet100_mlp_6_64_proj-cat_all_aug_w_color_diff_lr_0.03_seed_0"

## MOCO

# official
# lr 0.03
# baseline
# mlp 6 64 0.03
# mlp 6 64 col diff 0.03

## SIMSIAM

# augself
PATH1="simsiam-resnet50-imagenet100_augself_OFFICIAL"
# baseline
PATH2="simsiam-resnet50-imagenet100_baseline_OFFICIAL"
# mlp 6 64
PATH3="simsiam-resnet50-imagenet100_mlp_6_64_proj-cat_all_aug_lr_0.05_seed_0"
# mlp 6 64 col diff
PATH4="simsiam-resnet50-imagenet100_mlp_6_64_proj-cat_all_aug_w_color_diff_lr_0.05_seed_0"

python pca.py \
    --model $BACKBONE --pretrain-data $pretrain --ckpt $MODEL_PATH --dataset "cifar100" --datadir $CIFAR100_PATH --batch-size 1000 --k 1000
