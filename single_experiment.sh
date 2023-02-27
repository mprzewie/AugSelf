EXP_NAME="${FRAMEWORK}-${BACKBONE}-${PRETRAIN_DATASET}_${AUG_TREATMENT}_${AUG_NN_DEPTH}_${AUG_NN_WIDTH}_${AUG_INJ}_${SUFFIX}"
OUT_DIR="${RES_DIR}${EXP_NAME}"


CUDA_VISIBLE_DEVICES=$CUDA python pretrain_cond.py \
          --logdir $OUT_DIR \
          --resume $RESUME \
          --framework $FRAMEWORK \
          --dataset $PRETRAIN_DATASET \
          --datadir ${datadirs["$PRETRAIN_DATASET"]} \
          --model $BACKBONE \
          --batch-size 256 \
          --max-epochs $MAX_EPOCHS \
          --aug-treatment $AUG_TREATMENT \
          --aug-nn-depth $AUG_NN_DEPTH --aug-nn-width $AUG_NN_WIDTH \
          --aug-inj-type $AUG_INJ $EXTRA_ARGS

source "${SRC_DIR}single_eval.sh"