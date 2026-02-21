#!/bin/bash
set -e

SUBMISSION_DIR="/workspace/submissions"
CHECKPOINT="${1:-/workspace/experiments/best.pt}"
MODEL="${2:-resnet50}"

echo "Generating submission with checkpoint: $CHECKPOINT"

python /workspace/src/inference.py \
    --checkpoint "$CHECKPOINT" \
    --model "$MODEL" \
    --output "$SUBMISSION_DIR/submission.csv"

echo "Validating submission format..."
head -n 5 "$SUBMISSION_DIR/submission.csv"
wc -l "$SUBMISSION_DIR/submission.csv"

echo "Done. Submission saved to $SUBMISSION_DIR/submission.csv"
