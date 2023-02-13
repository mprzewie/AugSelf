source "${SRC_DIR}ovh_setup_datadirs.sh"

#### linear evaluation
for DS in "cifar10" "cifar100" "sun397" "caltech_101" "food101" "dtd" "pets" "cars" "aircraft" "flowers";
do
  {
    DS_DIR="${datadirs["$DS"]}"
    DS_METRIC="${metrics["$DS"]}"
    DS_OUT_FILE="linear_${DS}_${DS_METRIC}.txt"

    CKPT_SEQ=$(for E in `seq $EVAL_FREQ $EVAL_FREQ $MAX_EPOCHS`; do echo "${OUT_DIR}/ckpt-${E}.pth"; done)

    CUDA_VISIBLE_DEVICES=$CUDA python transfer_linear_eval.py \
      --pretrain-data $PRETRAIN_DATASET \
      --ckpt $CKPT_SEQ \
      --model $BACKBONE \
      --dataset $DS \
      --datadir $DS_DIR \
      --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"
  }
done


#
#
#### few-shot evaluation
#
#for DS in "plant_disease"; # "fc100" "cub200"
#do
#  for N in 5;
#  do
#    for K in 1 5;
#    do
#      DS_OUT_FILE="few-shot_${DS}_${N}_${K}.txt"
#      DS_DIR="${datadirs["$DS"]}"
#
#      CUDA_VISIBLE_DEVICES=0 python transfer_few_shot.py \
#        --pretrain-data $PRETRAIN_DATASET \
#        --ckpt "${OUT_DIR}/ckpt-${EVAL_EPOCHS}.pth" \
#        --model $BACKBONE \
#        --dataset $DS \
#        --datadir $DS_DIR \
#        --N 5 --K $K | tee "${OUT_DIR}/${DS_OUT_FILE}"
#    done
#  done
#
#done
#
##### linear (LOOC-like) evaluation
#
#for DS in "cub200"; #"imagenet100";
#do
#  CUDA_VISIBLE_DEVICES=0 python transfer_looc_like.py \
#    -a $BACKBONE --lr 30.0 \
#    --batch-size 256 \
#    --dist-url 'tcp://localhost:10002' --world-size 1 --rank 0 --epochs 200 --schedule 120 160 \
#    --workers 4 \
#    --pretrained "${OUT_DIR}/ckpt-${EVAL_EPOCHS}.pth" \
#    --dataset-name $DS ${datadirs["$DS"]}
#done