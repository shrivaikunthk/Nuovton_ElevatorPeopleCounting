#!/usr/bin/env bash
# Run all FOMO finetuning experiments sequentially.
#
# Each experiment trains a model and writes artifacts under
# runs/<name>/ (PyTorch) or keras_fomo/runs/<name>/ (Keras), then sweeps
# thresholds and writes sweep_*.json + sweep_*.csv next to the checkpoint.
#
# All variants kicked off here:
#   PyTorch:
#     pt_v1_lower_posweight   -- pos_weight=20 (vs 100 baseline)
#     pt_v2_focal             -- focal loss
#     pt_v3_grid12            -- grid_size=12
#     pt_v4_full              -- pos_weight=20 + jitter + count_aux + grid12
#   Keras:
#     ks_v1_jitter            -- brightness/contrast augmentation
#     ks_v2_count_aux         -- count-regression aux loss
#     ks_v3_grid12            -- grid_size=12
#     ks_v4_full              -- jitter + count_aux + grid12
#
# Then runs sweep_threshold_fomo.py for every checkpoint and finally calls
# compare_fomo_runs.py to print the leaderboard.
#
# Usage:
#   bash scripts/run_fomo_experiments.sh                  # full run (~12 hrs CPU)
#   EPOCHS=10 bash scripts/run_fomo_experiments.sh        # quick smoke run
#   ONLY="pt_v4_full ks_v4_full" bash scripts/run_fomo_experiments.sh

set -euo pipefail

EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-32}"
DATASET="${DATASET:-overhead-person-detection}"
ONLY="${ONLY:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

run_if_selected() {
    local name="$1"
    if [[ -z "$ONLY" || " $ONLY " == *" $name "* ]]; then
        echo ""
        echo "=========================================="
        echo "  $name"
        echo "=========================================="
        return 0
    fi
    return 1
}

# ---------------- PyTorch experiments ----------------

if run_if_selected pt_v1_lower_posweight; then
    python scripts/train_fomo.py \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --pos-weight 20 \
        --output-dir runs/pt_v1_lower_posweight \
        2>&1 | tee logs/pt_v1_lower_posweight.log
fi

if run_if_selected pt_v2_focal; then
    python scripts/train_fomo.py \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --focal --pos-weight 1 \
        --output-dir runs/pt_v2_focal \
        2>&1 | tee logs/pt_v2_focal.log
fi

if run_if_selected pt_v3_grid12; then
    python scripts/train_fomo.py \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --grid-size 12 --pos-weight 20 \
        --output-dir runs/pt_v3_grid12 \
        2>&1 | tee logs/pt_v3_grid12.log
fi

if run_if_selected pt_v4_full; then
    python scripts/train_fomo.py \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --grid-size 12 --pos-weight 20 \
        --brightness-jitter 0.2 --contrast-jitter 0.2 \
        --count-aux-weight 0.1 \
        --output-dir runs/pt_v4_full \
        2>&1 | tee logs/pt_v4_full.log
fi

# ---------------- Keras experiments ----------------

if run_if_selected ks_v1_jitter; then
    python -m keras_fomo.train \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --brightness-jitter 0.2 --contrast-jitter 0.2 \
        --output-dir keras_fomo/runs/ks_v1_jitter \
        2>&1 | tee logs/ks_v1_jitter.log
fi

if run_if_selected ks_v2_count_aux; then
    python -m keras_fomo.train \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --count-aux-weight 0.1 \
        --output-dir keras_fomo/runs/ks_v2_count_aux \
        2>&1 | tee logs/ks_v2_count_aux.log
fi

if run_if_selected ks_v3_grid12; then
    python -m keras_fomo.train \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --grid-size 12 \
        --output-dir keras_fomo/runs/ks_v3_grid12 \
        2>&1 | tee logs/ks_v3_grid12.log
fi

if run_if_selected ks_v4_full; then
    python -m keras_fomo.train \
        --dataset-root "$DATASET" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --grid-size 12 \
        --brightness-jitter 0.2 --contrast-jitter 0.2 \
        --count-aux-weight 0.1 \
        --output-dir keras_fomo/runs/ks_v4_full \
        2>&1 | tee logs/ks_v4_full.log
fi

# ---------------- Threshold sweeps for every checkpoint ----------------

echo ""
echo "=========================================="
echo "  Threshold sweeps"
echo "=========================================="

# Include the original baseline runs too if they exist.
declare -A SWEEPS=(
    [baseline_pt]="runs/fomo/best.pt"
    [pt_v1_lower_posweight]="runs/pt_v1_lower_posweight/best.pt"
    [pt_v2_focal]="runs/pt_v2_focal/best.pt"
    [pt_v3_grid12]="runs/pt_v3_grid12/best.pt"
    [pt_v4_full]="runs/pt_v4_full/best.pt"
    [baseline_ks]="keras_fomo/runs/fomo/best.keras"
    [ks_v1_jitter]="keras_fomo/runs/ks_v1_jitter/best.keras"
    [ks_v2_count_aux]="keras_fomo/runs/ks_v2_count_aux/best.keras"
    [ks_v3_grid12]="keras_fomo/runs/ks_v3_grid12/best.keras"
    [ks_v4_full]="keras_fomo/runs/ks_v4_full/best.keras"
)

for name in "${!SWEEPS[@]}"; do
    model="${SWEEPS[$name]}"
    if [[ -f "$model" ]]; then
        echo "-- sweep $name ($model)"
        python scripts/sweep_threshold_fomo.py \
            --dataset-root "$DATASET" --model "$model" \
            2>&1 | tee "logs/sweep_${name}.log" || true
    else
        echo "-- skip $name (no checkpoint at $model)"
    fi
done

# ---------------- Final comparison ----------------

echo ""
echo "=========================================="
echo "  Leaderboard"
echo "=========================================="
python scripts/compare_fomo_runs.py
