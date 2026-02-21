#!/bin/bash

FREQ_PID=313834
DINO_PID=313832

echo "=== Waiting for FreqClassifier (PID $FREQ_PID) to finish ==="
while kill -0 $FREQ_PID 2>/dev/null; do
  sleep 30
done
echo "FreqClassifier finished at $(date)"

echo "=== Starting CLIP-RINE v5 training ==="
cd /root/NTIRE/src
python train.py \
  --model_type rine \
  --backbone ViT-L/14 \
  --data_root /root/autodl-tmp/NTIRE_dataset \
  --shards 0,1,2 \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_hooks 12 \
  --label_smoothing 0.05 \
  --output_dir /root/autodl-tmp/experiments/rine_v5 \
  2>&1 | tee /root/autodl-tmp/experiments/rine_v5_train.log &

RINE_PID=$!
echo "RINE PID: $RINE_PID"

# Wait for DINOv3
echo "Waiting for DINOv3 (PID $DINO_PID)..."
while kill -0 $DINO_PID 2>/dev/null; do
  sleep 30
done
echo "DINOv3 finished at $(date)"

# Wait for RINE
echo "Waiting for RINE (PID $RINE_PID)..."
wait $RINE_PID
echo "RINE finished at $(date)"

# Cancel any pending shutdown from original script
shutdown -c 2>/dev/null

echo "=== All 3 models trained. Shutting down ==="
/usr/bin/shutdown -h +1
