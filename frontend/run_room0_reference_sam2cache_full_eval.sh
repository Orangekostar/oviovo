#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/frontend_proposals/room0_sam2_cache}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reference_20260422_room0_ovopro_sam2cache_full_eval}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/outputs/${EXPERIMENT_NAME}.log}"

conda run -n oviovo python run_room0_full_eval.py \
  --experiment-name "$EXPERIMENT_NAME" \
  --proposal-backend precomputed \
  --proposal-cache-dir "$CACHE_DIR" \
  2>&1 | tee "$LOG_PATH"
