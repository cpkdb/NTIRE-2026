#!/bin/bash
set -e

cd /root/NTIRE/src

echo "=== Starting parallel retraining ==="
echo "Start time: $(date)"

# DINOv3+LoRA: 15 epochs, from scratch, new augmentations already in transforms.py
python train.py \
  --model_type dinov3 \
  --backbone dinov2_vitl14_reg \
  --data_root /root/autodl-tmp/NTIRE_dataset \
  --shards 0,1,2 \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_hooks 12 \
  --lora_layers 6 \
  --label_smoothing 0.05 \
  --output_dir /root/autodl-tmp/experiments/dinov3_v2_lora \
  2>&1 | tee /root/autodl-tmp/experiments/dinov3_v2_lora_train.log &

PID_DINO=$!

# FreqClassifier: 20 epochs, from scratch
python train.py \
  --model_type freq \
  --backbone dummy \
  --data_root /root/autodl-tmp/NTIRE_dataset \
  --shards 0,1,2 \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4 \
  --label_smoothing 0.05 \
  --output_dir /root/autodl-tmp/experiments/freq_v2 \
  2>&1 | tee /root/autodl-tmp/experiments/freq_v2_train.log &

PID_FREQ=$!

echo "DINOv3 PID: $PID_DINO"
echo "Freq PID: $PID_FREQ"

# Wait for both to finish
wait $PID_DINO
DINO_EXIT=$?
echo "DINOv3 finished with exit code: $DINO_EXIT"

wait $PID_FREQ
FREQ_EXIT=$?
echo "Freq finished with exit code: $FREQ_EXIT"

echo "=== All training complete ==="
echo "End time: $(date)"

# Shutdown handled by run_rine_after_freq.sh relay script
echo "DINOv3+Freq done. RINE relay script handles shutdown."
