#!/bin/bash
# Wait until RINE training starts (after freq finishes)
FREQ_PID=313834
while kill -0 $FREQ_PID 2>/dev/null; do
  sleep 30
done
echo "Freq done, RINE should start soon. Activating shutdown guard."
sleep 30

# Keep canceling any shutdown while any train.py is still running
while pgrep -f "python train.py" > /dev/null 2>&1; do
  shutdown -c 2>/dev/null
  sleep 20
done
echo "All training processes done. Guard deactivated at $(date)"
