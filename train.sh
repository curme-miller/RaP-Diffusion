#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Training parameters
DATA_ROOT="/home/v4-5090/zlx/data/CtM"
SAVE_DIR="/home/v4-5090/zlx/data/CtM"
CONFIG_FILE="config/config.yaml"
BATCH_SIZE=8
LEARNING_RATE=5e-5
ITERATIONS=60000

python train.py \
    --save_dir ${SAVE_DIR} \
    --cfg_path ${CONFIG_FILE} \
    --data_root ${DATA_ROOT} \
    --gpu_ids ${CUDA_VISIBLE_DEVICES} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --iterations ${ITERATIONS}

echo ""
echo "🎉 Training completed!"
echo "📊 Check training log: ./training.log"
echo "💾 Checkpoints saved in: ./checkpoints/$(basename ${DATA_ROOT})/"
echo "🖼️  Validation images saved in: ./validation_images/$(basename ${DATA_ROOT})/"