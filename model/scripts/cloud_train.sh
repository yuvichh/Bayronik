#!/usr/bin/env bash
# Train the N-body -> hydro emulator on a rented CUDA machine (vast.ai, Colab, ...).
#
# The machine fetches the CAMELS maps itself (7.9 GB, straight from Flatiron),
# so nothing large is uploaded from a laptop. Run from the model/ directory:
#
#   BUDGET_MIN=50 bash scripts/cloud_train.sh            # fit the run to 50 minutes
#   EPOCHS=40 BUDGET_MIN=0 bash scripts/cloud_train.sh   # fixed 40 epochs
#   RESUME=1 bash scripts/cloud_train.sh                 # continue from last.pt
#
# Weights land in weights/$RUN_NAME/{best.pt,last.pt,log.jsonl,split.json}.
set -euo pipefail

BUDGET_MIN=${BUDGET_MIN:-50}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-16}
RUN_NAME=${RUN_NAME:-ufno_v2_Mtot_Nbody_IllustrisTNG}
DATA_DIR=${DATA_DIR:-data}
WITH_CV=${WITH_CV:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

CMD=https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data

fetch() {
    local url=$1 out="$DATA_DIR/$(basename "$1")"
    if [ ! -s "$out" ]; then
        curl -fL --retry 5 --retry-delay 3 -C - -o "$out.part" "$url"
        mv "$out.part" "$out"
    fi
}

mkdir -p "$DATA_DIR"
start=$(date +%s)
fetch "$CMD/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy" &
fetch "$CMD/Nbody/L25n256/Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.00.npy" &
fetch "$CMD/IllustrisTNG/params_LH_IllustrisTNG.txt" &
if [ "$WITH_CV" = 1 ]; then
    fetch "$CMD/IllustrisTNG/Maps_Mtot_IllustrisTNG_CV_z=0.00.npy" &
    fetch "$CMD/Nbody/L25n256/Maps_Mtot_Nbody_IllustrisTNG_CV_z=0.00.npy" &
fi
wait
echo "Data ready in $(( $(date +%s) - start ))s"
ls -la "$DATA_DIR"

python -c "import torch" 2>/dev/null || pip install -q torch
pip install -q numpy einops
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

RESUME_ARG=""
if [ "${RESUME:-0}" = 1 ] && [ -f "weights/$RUN_NAME/last.pt" ]; then
    RESUME_ARG="--resume weights/$RUN_NAME/last.pt"
fi

# shellcheck disable=SC2086
python train.py --data-dir "$DATA_DIR" --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" --time-budget-min "$BUDGET_MIN" --batch-size "$BATCH_SIZE" \
    $RESUME_ARG $EXTRA_ARGS

if [ "$WITH_CV" = 1 ]; then
    pip install -q matplotlib
    python benchmarks/nbody2hydro.py --weights "weights/$RUN_NAME/best.pt" --data-dir "$DATA_DIR" \
        --out "weights/$RUN_NAME/benchmark"
fi
