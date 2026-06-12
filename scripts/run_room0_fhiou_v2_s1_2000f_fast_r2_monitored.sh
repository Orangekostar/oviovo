#!/usr/bin/env bash
set -u

cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates

experiment=20260601_room0_fast_high_iou_v2_stride1_2000f_fast_r2
run_root=outputs/tmp_validation/${experiment}
log=outputs/tmp_validation/${experiment}.log
status=outputs/tmp_validation/${experiment}.status
wait_log=outputs/tmp_validation/${experiment}.resource_wait.log
resource_log=outputs/tmp_validation/${experiment}.resource_monitor.tsv
frame_metrics=${run_root}/room0/frame_metrics.jsonl

gpu_compute_mem_mb() {
  nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}'
}

gpu_util_pct() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 \
    | awk '{print $1 + 0}'
}

echo "experiment=${experiment}" > "${wait_log}"
echo "queued_at=$(date --iso-8601=seconds)" >> "${wait_log}"
echo "waiting_for_compute_gpu_mem_mb<=512_and_gpu_util_pct<=20" >> "${wait_log}"

while true; do
  compute_mem=$(gpu_compute_mem_mb)
  util=$(gpu_util_pct)
  echo "wait_check=$(date --iso-8601=seconds) compute_mem_mb=${compute_mem} gpu_util_pct=${util}" >> "${wait_log}"
  if [ "${compute_mem}" -le 512 ] && [ "${util}" -le 20 ]; then
    break
  fi
  sleep 60
done

echo "started_after_wait_at=$(date --iso-8601=seconds)" >> "${wait_log}"

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
    --fast-eval &
  run_pid=$!

  {
    echo -e "timestamp\tpid\trss_kb\telapsed\tframe_rows\tgpu_compute_mem_mb\tgpu_util_pct\tmem_available_mb\tswap_free_mb"
    while kill -0 "${run_pid}" 2>/dev/null; do
      rss=$(ps -p "${run_pid}" -o rss= 2>/dev/null | awk '{print $1 + 0}')
      elapsed=$(ps -p "${run_pid}" -o etime= 2>/dev/null | awk '{print $1}')
      rows=0
      if [ -f "${frame_metrics}" ]; then
        rows=$(wc -l < "${frame_metrics}")
      fi
      compute_mem=$(gpu_compute_mem_mb)
      util=$(gpu_util_pct)
      read -r mem_available swap_free < <(free -m | awk '/Mem:/ {mem=$7} /Swap:/ {swap=$4} END {print mem, swap}')
      echo -e "$(date --iso-8601=seconds)\t${run_pid}\t${rss}\t${elapsed}\t${rows}\t${compute_mem}\t${util}\t${mem_available}\t${swap_free}"
      sleep 30
    done
  } > "${resource_log}" &
  monitor_pid=$!

  wait "${run_pid}"
  rc=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_status=${rc}"
  echo "exit_status=${rc}" > "${status}"
} > "${log}" 2>&1
