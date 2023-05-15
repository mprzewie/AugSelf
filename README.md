# Augmentation-aware Self-Supervised-Learning with Guided Projector

Under review.


**TL;DR:**  We condition the projector of self-supervised models with augmentation information and demonstrate that this improves their performance during transfer learning.
## Dependencies

```bash
conda create -n CASSLE python=3.8 pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.1 ignite -c pytorch
conda activate AugSelf
pip install scipy tensorboard kornia==0.4.1 sklearn wandb
```


## Pretraining

We provide a command for pretraining MoCo-v2 + CASSLE. To train the Baseline model, replace the `--aug-inj-type` option to `proj-none`. To train [AugSelf](https://arxiv.org/abs/2111.09613), use `--aug-inj-type proj-none --ss-crop 0.5 --ss-color 0.5`. 
For using other frameworks like SimCLR, use the `--framework` option.


```bash
pretrain_cond.py  \
  --logdir $LOGDIR \
  --framework moco --dataset imagenet100 --datadir $IMGENET_100_FOLDER \
  --model resnet50 --batch-size 256 --max-epochs 500 \
  --aug-treatment mlp --aug-nn-depth 6 --aug-nn-width 64 --aug-inj-type proj-cat \
  --aug-cond crop color color_diff blur grayscale \
  --base-lr 0.03 --wd 1e-4 --ckpt-freq 50 --eval-freq 50 --num-workers 16 --seed 1 --distributed
```

## Evaluation

Our main evaluation setups are linear evaluation on fine-grained classification datasets (Table 1).
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

