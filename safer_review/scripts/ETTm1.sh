#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}
MODEL=iTransformer
SEQ_LEN=720
LABEL_LEN=48
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
RUN_TAG=${RUN_TAG:-delayed_causal_regularizer}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
use_safer=${USE_SAFER:-1}
use_shift_safe=${USE_SHIFT_SAFE:-1}
use_instant_causal=${USE_INSTANT_CAUSAL:-1}
use_delayed_regularizer=${USE_DELAYED_CAUSAL_REGULARIZER:-1}
safer_alpha=${SAFER_ALPHA:-1.0}
for PRED_LEN in $PRED_LENS; do
  MODEL_ID="ETTm1_${SEQ_LEN}_${PRED_LEN}_${RUN_TAG}"
  echo "=================================================================================="
  echo "dataset=ETTm1 pred_len=${PRED_LEN} delayed_causal_regularizer=${use_delayed_regularizer}"
  echo "=================================================================================="
  $PYTHON -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id "$MODEL_ID" \
    --model $MODEL \
    --data ETTm1 \
    --features M \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des Exp \
    --d_model 32 \
    --d_ff 32 \
    --batch_size $BATCH_SIZE \
    --train_epochs $TRAIN_EPOCHS \
    --patience 3 \
    --learning_rate 0.0005 \
    --dropout 0.2 \
    --use_safer $use_safer \
    --use_shift_safe $use_shift_safe \
    --use_instant_causal $use_instant_causal \
    --safer_alpha $safer_alpha \
    --use_delayed_causal_regularizer $use_delayed_regularizer \
    --itr 1
done
