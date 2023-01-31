EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_${AUG_TREATMENT}_${AUG_NN_DEPTH}_${AUG_NN_WIDTH}_${AUG_INJ}_${SUFFIX}"
OUT_DIR="${BASE_DIR}/${EXP_NAME}"


# python pretrain_cond.py \
#          --logdir $OUT_DIR \
#          --framework $FRAMEWORK \
#          --dataset $PRETRAIN_DATASET \
#          --datadir ${datadirs["$PRETRAIN_DATASET"]} \
#          --model $BACKBONE \
#          --batch-size 256 \
#          --max-epochs $MAX_EPOCHS \
#          --aug-treatment $AUG_TREATMENT \
#          --aug-nn-depth $AUG_NN_DEPTH --aug-nn-width $AUG_NN_WIDTH \
#          --aug-inj-type $AUG_INJ $EXTRA_ARGS

source single_eval.sh

