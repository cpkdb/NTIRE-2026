#!/bin/bash
DINO_PID=313832
FREQ_PID=313834
RINE_PID=323611

echo "Monitoring 3 training processes: DINOv3=$DINO_PID, Freq=$FREQ_PID, RINE=$RINE_PID"

# Cancel any shutdown from original script while training is active
while kill -0 $DINO_PID 2>/dev/null || kill -0 $FREQ_PID 2>/dev/null || kill -0 $RINE_PID 2>/dev/null; do
  shutdown -c 2>/dev/null
  sleep 30
done

echo "All 3 training processes completed at $(date)"
/usr/bin/shutdown -h +1
