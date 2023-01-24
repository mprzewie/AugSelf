EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_${AUG_TREATMENT}_${AUG_NN_DEPTH}_${AUG_NN_WIDTH}_${AUG_INJ}_${SUFFIX}"
OUT_DIR="${BASE_DIR}/${EXP_NAME}"


CUDA_VISIBLE_DEVICES=0 python pretrain_cond.py \
          --logdir $OUT_DIR \
          --framework $FRAMEWORK \
          --dataset $PRETRAIN_DATASET \
          --datadir ${datadirs["$PRETRAIN_DATASET"]} \
          --model $BACKBONE \
          --batch-size 256 \
          --max-epochs $MAX_EPOCHS \
          --aug-treatment $AUG_TREATMENT \
          --aug-nn-depth $AUG_NN_DEPTH --aug-nn-width $AUG_NN_WIDTH \
          --aug-inj-type $AUG_INJ $EXTRA_ARGS

#### linear evaluation

for DS in ${!datadirs[@]};
do
  DS_DIR="${datadirs["$DS"]}"
  DS_METRIC="${metrics["$DS"]}"


  DS_OUT_FILE="linear_${DS}_${DS_METRIC}.txt"

  CUDA_VISIBLE_DEVICES=0 python transfer_linear_eval.py \
    --pretrain-data stl10 \
    --ckpt ${OUT_DIR}/ckpt-200.pth \
    --model $BACKBONE \
    --dataset $DS \
    --datadir $DS_DIR \
    --metric $DS_METRIC 2>&1 | tee "${OUT_DIR}/${DS_OUT_FILE}"
done