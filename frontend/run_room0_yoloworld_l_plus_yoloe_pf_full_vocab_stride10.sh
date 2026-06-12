#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/frontend_proposals/room0_sam2_cache_full_midrecall_real_v1_20260427}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-yoloworld_l_plus_yoloe_pf_realcache_room0_fullvocab_stride10_200f_20260506}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/outputs/tmp_validation/${EXPERIMENT_NAME}.log}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/outputs/tmp_validation/${EXPERIMENT_NAME}.yaml}"
YOLOWORLD_MODEL_PATH="${YOLOWORLD_MODEL_PATH:-/home/phl/vv/DualMapV1/model/yolov8l-world.pt}"
YOLOWORLD_PYTHON="${YOLOWORLD_PYTHON:-/home/phl/anaconda3/envs/dualmap/bin/python}"
YOLOE_PYTHON="${YOLOE_PYTHON:-/home/phl/anaconda3/envs/oviovo/bin/python}"
YOLOE_REPO_ROOT="${YOLOE_REPO_ROOT:-/home/phl/vv/yoloe_repo_probe}"
YOLOE_CHECKPOINT_PATH="${YOLOE_CHECKPOINT_PATH:-/home/phl/vv/yoloe_repo_probe/pretrain/yoloe-v8s-seg-pf.pt}"
PRIMARY_DEVICE="${PRIMARY_DEVICE:-cuda:0}"
SUPPLEMENTAL_DEVICE="${SUPPLEMENTAL_DEVICE:-cuda:0}"

python3 frontend/build_room0_full_vocab_config.py \
  --base-config "$ROOT_DIR/configs/midrecall_local_memory_boost.yaml" \
  --output-config "$CONFIG_PATH" \
  --yoloworld-model-path "$YOLOWORLD_MODEL_PATH" \
  --yoloworld-python "$YOLOWORLD_PYTHON" \
  --yoloe-python "$YOLOE_PYTHON" \
  --yoloe-repo-root "$YOLOE_REPO_ROOT" \
  --yoloe-checkpoint-path "$YOLOE_CHECKPOINT_PATH" \
  --primary-device "$PRIMARY_DEVICE" \
  --supplemental-device "$SUPPLEMENTAL_DEVICE"

python3 run_room0_full_eval.py \
  --config-path "$CONFIG_PATH" \
  --num-frames 0 \
  --frame-stride 10 \
  --proposal-backend precomputed \
  --proposal-cache-dir "$CACHE_DIR" \
  --output-root "$ROOT_DIR/outputs/tmp_validation" \
  --experiment-name "$EXPERIMENT_NAME" \
  2>&1 | tee "$LOG_PATH"
