#!/bin/bash

cd /root/NTIRE/src
echo "=== All 3 models parallel training ==="
echo "Start: $(date)"

python train.py \
  --model_type dinov3 --backbone dinov2_vitl14_reg \
  --data_root /root/autodl-tmp/NTIRE_dataset --shards 0,1,2 \
  --epochs 15 --batch_size 32 --lr 1e-4 \
  --num_hooks 12 --lora_layers 6 --label_smoothing 0.05 \
  --output_dir /root/autodl-tmp/experiments/dinov3_v2_lora \
  > /root/autodl-tmp/experiments/dinov3_v2_lora_train.log 2>&1 &
P1=$!

python train.py \
  --model_type freq --backbone dummy \
  --data_root /root/autodl-tmp/NTIRE_dataset --shards 0,1,2 \
  --epochs 20 --batch_size 64 --lr 1e-4 \
  --label_smoothing 0.05 \
  --output_dir /root/autodl-tmp/experiments/freq_v2 \
  > /root/autodl-tmp/experiments/freq_v2_train.log 2>&1 &
P2=$!

python train.py \
  --model_type rine --backbone ViT-L/14 \
  --data_root /root/autodl-tmp/NTIRE_dataset --shards 0,1,2 \
  --epochs 15 --batch_size 32 --lr 1e-4 \
  --num_hooks 12 --label_smoothing 0.05 \
  --output_dir /root/autodl-tmp/experiments/rine_v5 \
  > /root/autodl-tmp/experiments/rine_v5_train.log 2>&1 &
P3=$!

echo "PIDs: DINOv3=$P1 Freq=$P2 RINE=$P3"

# Wait for ALL three
wait $P1; echo "DINOv3 done (exit=$?) at $(date)"
wait $P2; echo "Freq done (exit=$?) at $(date)"
wait $P3; echo "RINE done (exit=$?) at $(date)"

echo "=== ALL COMPLETE at $(date) ==="
/usr/bin/shutdown -h now
