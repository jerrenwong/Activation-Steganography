#!/bin/bash
# Quick-start script for overnight experiment run
# Usage: bash start.sh
#
# What this does:
#   1. Verifies GPU access
#   2. Installs Python dependencies
#   3. Disables wandb (set WANDB_API_KEY first if you want logging)
#   4. Runs the full pipeline: probe sweep → adversarial training → evaluation
#   5. Logs everything to results/run_all.log

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Checking GPU ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB')"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== Setting up ==="
mkdir -p results checkpoints

# Disable wandb unless API key is set
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "No WANDB_API_KEY found, disabling wandb"
    export WANDB_MODE=disabled
fi

echo "=== Starting full pipeline (probe sweep → training → eval) ==="
echo "Logs: results/run_all.log"
echo "Checkpoints: checkpoints/"
echo "Results: results/*.json"
echo ""
echo "Estimated runtime: 6-10 hours on A100/RTX-5090"
echo "Started at: $(date)"
echo ""

nohup python3 -m scripts.run_all --config configs/default.yaml > results/stdout.log 2>&1 &
PID=$!
echo "Running in background (PID: $PID)"
echo "$PID" > results/run.pid
echo ""
echo "Monitor progress:"
echo "  tail -f results/run_all.log"
echo "  tail -f results/stdout.log"
echo ""
echo "Check if still running:"
echo "  kill -0 \$(cat results/run.pid) 2>/dev/null && echo 'Running' || echo 'Done'"
