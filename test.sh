#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

for d in SmM
do

  CONFIG_FILE="config.yaml"
  CHECKPOINT_PATH="./checkpoints/${d}/best_model.pth"
  TEST_DATA_PATH="/home/v4-5090/zlx/data/${d}/test"
  OUTPUT_PATH="./test_img/${d}"

  python test.py \
      --cfg_path ${CONFIG_FILE} \
      --checkpoint_path ${CHECKPOINT_PATH} \
      --test_data_path ${TEST_DATA_PATH} \
      --output_path ${OUTPUT_PATH} \
      --gpu_ids ${CUDA_VISIBLE_DEVICES}

done