# OVIOVO 0526 Superpowers Docker Restore

Date: 2026-06-12

## Active container

- Container: `ww-ai`
- Image: `ai-conda:cuda12.4-iad`
- Workdir: `/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates`
- Previous container preserved as: `ww-ai-base-20260612`

Mounted into `ww-ai`:

```text
/home/ww/vv                         -> /home/ww/vv
/home/ww/.config/superpowers         -> /home/ww/.config/superpowers
/home/ww/backups                     -> /home/ww/backups (read-only)
/home/ww/ww-ai/projects              -> /workspace
/home/ww/ww-ai/data                  -> /data
/home/ww/ww-ai/logs                  -> /logs
/home/ww/ww-ai/config                -> /config
```

## 0526 experiment location

Primary Superpowers worktree:

```text
/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
```

Useful 0526 outputs:

```text
outputs/tmp_validation/
  20260526_room0_observation_first_2b830e1_stride1_2000f/
    replica/results.json
    room0/run_report.md
    room0/exports/room0_instance_map.ply
    room0/exports/room0_instance_map_dense_surface.ply
    room0/exports/room0_dense_geometry_instance_projected.ply
  20260526_room0_semantic_vote_weak_structure_stride10_200f/
    replica/results.json
    room0/run_report.md
    room0/exports/
```

Confirmed metrics:

```text
20260526_room0_observation_first_2b830e1_stride1_2000f:
  mIoU  = 0.6049961456525983
  f-mIoU = 0.75312622297826

20260526_room0_semantic_vote_weak_structure_stride10_200f:
  mIoU  = 0.5457083375710071
  f-mIoU = 0.6307995580918061
```

## Important finding

The 0526 high-coverage run was not the later SED frontend. The logs show:

```text
backend: precomputed
config:  configs/room0_surface_gate_4090.yaml
```

It relied on precomputed SAM proposals plus YOLOWorld/YOLOE anchor frontend.
The later `vv/oviovo0526_sed_frontend` directory is a small SED-oriented snapshot,
not the complete 0526 Superpowers run environment.

## Exact rerun blockers

The Docker path restoration is done, but an exact rerun still needs these missing inputs:

```text
/home/ww/vv/dataset/Replica/room0
/home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2
/home/ww/vv/DualMapV1/model/yolov8l-world.pt
/home/ww/vv/yoloe_repo_probe/pretrain/yoloe-v8s-seg-pf.pt
/home/ww/vv/yoloe_repo_probe/pretrain/yoloe-v8s-seg.pt
```

The original host Python path `/home/ww/miniconda3/envs/oviovo/bin/python` is
also not present on this machine. The new `ww-ai` has conda envs under
`/opt/conda/envs/`; `cird` is the closest starting point, but it still needs
at least `pyyaml`, `pytest`, `open3d`, `ultralytics`, and any project-specific
frontend dependencies before a full rerun.

## Check restored results in Docker

```bash
docker exec -it ww-ai bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
cat outputs/tmp_validation/20260526_room0_observation_first_2b830e1_stride1_2000f/replica/results.json
ls -lh outputs/tmp_validation/20260526_room0_observation_first_2b830e1_stride1_2000f/room0/exports
```

## Rerun template after missing inputs are restored

```bash
docker exec -it ww-ai bash
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates

/opt/conda/envs/cird/bin/python run_room0_full_eval.py \
  --config-path configs/room0_surface_gate_4090.yaml \
  --experiment-name 20260526_room0_semantic_vote_weak_structure_stride10_200f_rerun_wwai \
  --num-frames 200 \
  --frame-stride 10 \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --output-root outputs/tmp_validation \
  --proposal-backend precomputed \
  --proposal-cache-dir /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2 \
  --proposal-cache-manifest /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json
```
