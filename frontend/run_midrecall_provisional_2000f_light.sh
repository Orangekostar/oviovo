#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/frontend_proposals/room0_sam2_cache_full_midrecall_v1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-midrecall_provisional_2000f_light}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/outputs/tmp_validation/${EXPERIMENT_NAME}.log}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/oviovo_midrecall_provisional_full.yaml}"

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

python3 run_room0_full_eval.py \
  --config-path "$CONFIG_PATH" \
  --num-frames 0 \
  --proposal-backend precomputed \
  --proposal-cache-dir "$CACHE_DIR" \
  --output-root "$ROOT_DIR/outputs/tmp_validation" \
  --experiment-name "$EXPERIMENT_NAME" \
  --lightweight-benchmark
