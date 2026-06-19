#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/frontend_proposals/room0_sam2_cache_full_midrecall_v1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-room0_sam2_midrecall_cache_full_eval}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/outputs/${EXPERIMENT_NAME}.log}"

conda run -n oviovo python frontend/precompute_sam2_proposals_parallel.py \
  --dataset-root "$ROOT_DIR/data/input/Datasets/Replica/room0" \
  --output-dir "$CACHE_DIR" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --num-frames 0 \
  --sam2-version 2.1 \
  --sam2-encoder hiera_l \
  --sam-repo-root /home/phl/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path "$ROOT_DIR/data/input/sam_ckpts" \
  --points-per-side 24 \
  --max-proposals 96 \
  --confidence-threshold 0.0 \
  --min-mask-area 0 \
  --stability-score-th 0.92 \
  --nms-iou-th 0.8 \
  --min-mask-region-area 60 \
  --overwrite

conda run -n oviovo python run_room0_full_eval.py \
  --experiment-name "$EXPERIMENT_NAME" \
  --proposal-backend precomputed \
  --proposal-cache-dir "$CACHE_DIR" \
  2>&1 | tee "$LOG_PATH"
