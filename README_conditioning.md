# Improving Transferability of Representations via Augmentation-Aware Self-Supervision

Accepted to NeurIPS 2021

<p align="center">
<img width="762" alt="thumbnail" src="https://user-images.githubusercontent.com/4075389/138967888-29208bbe-d9e7-4bc7-b0b6-15ecbd5d277c.png">
</p>

**TL;DR:** Learning augmentation-aware information by conditioning on the encodings of two augmentations improves the transferability of representations. This is extention of AugSelf repo!

## Dependencies

```bash
conda create -n AugSelf python=3.8 pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 ignite -c pytorch
conda activate AugSelf
pip install scipy tensorboard kornia==0.4.1 sklearn

conda create -n AugSelfConidtioning python=3.8
conda activate AugSelfConditioning
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install scipy tensorboard kornia==0.4.1 sklearn
conda install -c conda-forge packaging
conda install -c conda-forge wandb
conda install ignite -c pytorch
```

## Checkpoints

We provide ImageNet100-pretrained models in [this Dropbox link](https://www.dropbox.com/sh/0hjts19ysxebmaa/AABB6bF3QQWdIOCh9vocwTGGa?dl=0).

## Pretraining

We here provide SimSiam+ConditioningMLP pretraining scripts. For training the baseline (i.e., no MLP component), remove `--ss-crop` and `--ss-color` options. For using other frameworks like SimCLR, use the `--framework` option.

### STL-10
```
script ovh_train_[***]_stl.sh
```

### ImageNet100

```
script ovh_train_[***]_imnet.sh
```

## Evaluation

```
script ovh_eval.sh
```

### linear evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python transfer_linear_eval.py \
    --pretrain-data imagenet100 \
    --ckpt CKPT \
    --model resnet50 \
    --dataset cifar10 \
    --datadir DATADIR \
    --metric top1
```

### few-shot

```bash
CUDA_VISIBLE_DEVICES=0 python transfer_few_shot.py \
    --pretrain-data imagenet100 \
    --ckpt CKPT \
    --model resnet50 \
    --dataset cub200 \
    --datadir DATADIR
```
