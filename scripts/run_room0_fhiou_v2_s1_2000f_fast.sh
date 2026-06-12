#!/usr/bin/env bash
set -u

cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates

experiment=20260601_room0_fast_high_iou_v2_stride1_2000f_fast
log=outputs/tmp_validation/${experiment}.log
status=outputs/tmp_validation/${experiment}.status

{
  echo "experiment=${experiment}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "worktree=$(pwd)"
  echo "dataset_root=/home/ww/vv/dataset/Replica/room0"
  /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py \
    --dataset-root /home/ww/vv/dataset/Replica/room0 \
    --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
    --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
    --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml \
    --output-root outputs/tmp_validation \
    --experiment-name "${experiment}" \
    --num-frames 2000 \
    --frame-stride 1 \
    --proposal-backend precomputed \
    --proposal-device cuda \
    --fast-eval
  rc=$?
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_status=${rc}"
  echo "exit_status=${rc}" > "${status}"
} > "${log}" 2>&1
