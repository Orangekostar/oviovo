# Dynamic Benchmark V4 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于已有 orbit 脚本生成 apt_0 2400 帧 12 变更动态基准数据集，含帧级 GT，验证 TSDF 驱动三级判决。

**Architecture:** 一次 Habitat 启动执行三阶段（800f before → delete+move → 800f mid → add → 800f after），NavMesh 校验防穿墙，Y-up 坐标放物。Pipeline 读取产出 frame_metrics，对比 event_log 评估。

**Tech Stack:** Python 3.9 (conda env habitat), Habitat Sim 0.3.3, NumPy, imageio, PyYAML, pytest

## Global Constraints

- 物体用 `NavMesh.snap_point()` 放置 → 绝不悬空
- 所有变更帧为 10 的整数倍: 800, 1600
- 相机用 `--traj-max-snap-dist 0.5` 防穿墙
- Y-up 坐标系统（与 orbit 脚本一致）
- OVIOVO pipeline 核心逻辑不改
- Habitat conda: `/home/ww/miniconda3/envs/habitat/bin/python`
- NVIDIA EGL: `LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib`
- YCB: `/home/ww/vv/habitat_data/objects/ycb/`
- ReplicaCAD: `/home/ww/vv/habitat_data/scene_datasets/replica_cad/`
- Traj: `/home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt`
- 输出: `outputs/dynamic_benchmark_v4/`

---

### Task 1: Modify orbit script for multi-object multi-phase generation

**Files:**
- Create: `/home/ww/vv/habitat-data-collector/scripts/20260227/generate_dynamic_benchmark.py`
- Based on: `/home/ww/vv/habitat-data-collector/scripts/20260227/replica_dynamic_object_orbit_capture.py`

**Interfaces:**
- Produces: script with `--traj-file`, `--output`, `--width`, `--height`, `--sensor-height`, `--min-clearance`, `--ycb-dir`, `--max-snap-dist` args
- Produces per run: `rgb/`, `depth/`, `event_log.json`, `camera_log.json`

- [ ] **Step 1: Copy base script**

```bash
cp /home/ww/vv/habitat-data-collector/scripts/20260227/replica_dynamic_object_orbit_capture.py \
   /home/ww/vv/habitat-data-collector/scripts/20260227/generate_dynamic_benchmark.py
echo "Copied"
```

- [ ] **Step 2: Rewrite args for multi-object**

Replace the `main()` argument parser with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dynamic benchmark with multi-object multi-phase support")
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--scene-dataset-config", required=True)
    parser.add_argument("--ycb-dir", required=True, help="YCB objects directory (e.g. .../objects/ycb/)")
    parser.add_argument("--traj-file", required=True, help="traj.txt (c2w format, Y-up)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sensor-height", type=float, default=1.5)
    parser.add_argument("--min-clearance", type=float, default=0.25)
    parser.add_argument("--max-snap-dist", type=float, default=0.5)
    parser.add_argument("--object-y-lift", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
```

- [ ] **Step 3: Add helper to load YCB and place object at NavMesh point**

Add this helper before `main()`:

```python
def load_ycb_templates(template_mgr, ycb_dir: Path) -> None:
    """Load all YCB object configs from directory tree."""
    for config_file in sorted(ycb_dir.rglob("*.object_config.json")):
        template_mgr.load_configs(str(config_file.parent))


def place_object_at_navmesh(
    rigid_mgr, template_mgr, keyword: str, pathfinder,
    target_pos: Optional[np.ndarray] = None,
    object_y_lift: float = 0.05,
) -> Optional[int]:
    """Place a YCB object at a NavMesh-validated point.
    
    If target_pos is None, picks a random navigable point.
    Returns object_id or None if placement fails.
    """
    handles = template_mgr.get_file_template_handles(keyword)
    if not handles:
        print(f"WARNING: no YCB template for keyword '{keyword}'")
        return None

    if target_pos is None:
        target_pos = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)

    snapped = pathfinder.snap_point(target_pos)
    if not np.isfinite(snapped).all():
        return None

    pos = np.array(snapped, dtype=np.float32)
    pos[1] += float(object_y_lift)

    obj = rigid_mgr.add_object_by_template_handle(handles[0])
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.translation = pos
    return int(obj.object_id)
```

- [ ] **Step 4: Add helper to place object on furniture surface**

```python
def place_object_on_surface(
    rigid_mgr, template_mgr, keyword: str,
    surface_pos: np.ndarray, surface_height_offset: float,
    object_y_lift: float = 0.05,
) -> Optional[int]:
    """Place a YCB object on top of a furniture surface.
    
    surface_pos: furniture world position (x, y, z) — Y is height.
    surface_height_offset: additional offset above furniture Y (e.g. 0.06 for tabletop).
    """
    handles = template_mgr.get_file_template_handles(keyword)
    if not handles:
        print(f"WARNING: no YCB template for keyword '{keyword}'")
        return None

    pos = np.array(surface_pos, dtype=np.float32)
    pos[1] += float(surface_height_offset) + float(object_y_lift)

    obj = rigid_mgr.add_object_by_template_handle(handles[0])
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.translation = pos
    return int(obj.object_id)
```

- [ ] **Step 5: Build the main generation logic (three-phase)**

Replace the main body (after `args = parser.parse_args()`) with:

```python
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.output)
    rgb_dir = out_root / "rgb"
    depth_dir = out_root / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectory
    traj_poses = load_traj_poses(
        traj_file=args.traj_file,
        stride=1, max_frames=0, pose_format="c2w",
    )
    print(f"Loaded {len(traj_poses)} trajectory poses")

    # Build simulator
    cfg = build_sim_cfg(
        scene_path=args.scene_path,
        scene_dataset_config=args.scene_dataset_config,
        width=args.width, height=args.height,
        sensor_height=args.sensor_height,
    )
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(0)
    pathfinder = sim.pathfinder
    template_mgr = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()

    ycb_dir = Path(args.ycb_dir)
    load_ycb_templates(template_mgr, ycb_dir)

    # Furniture positions (extracted from apt_0)
    furniture = {
        "table_01": np.array([0.41, 0.66, -0.17], dtype=np.float32),
        "table_02": np.array([0.90, 0.61, 2.41], dtype=np.float32),
        "sofa":     np.array([3.91, 0.40, 5.27], dtype=np.float32),
        "table_04": np.array([4.20, 0.44, 6.62], dtype=np.float32),
    }

    event_log = []
    all_object_ids = {}

    # === Phase 1: Place 8 objects, capture frames 0-799 ===
    # 4 on surfaces (bowl×2 on table_02, mug×2 on sofa)
    all_object_ids["bowl_0"] = place_object_on_surface(rigid_mgr, template_mgr, "bowl", furniture["table_02"], 0.06, args.object_y_lift)
    all_object_ids["bowl_1"] = place_object_on_surface(rigid_mgr, template_mgr, "bowl", furniture["table_02"], 0.06, args.object_y_lift)
    all_object_ids["mug_0"] = place_object_on_surface(rigid_mgr, template_mgr, "mug", furniture["sofa"], 0.40, args.object_y_lift)
    all_object_ids["mug_1"] = place_object_on_surface(rigid_mgr, template_mgr, "mug", furniture["sofa"], 0.40, args.object_y_lift)
    # 4 on floor (pitcher×2, foam_brick×2 — random NavMesh points away from furniture)
    all_object_ids["pitcher_0"] = place_object_at_navmesh(rigid_mgr, template_mgr, "pitcher", pathfinder, object_y_lift=args.object_y_lift)
    all_object_ids["pitcher_1"] = place_object_at_navmesh(rigid_mgr, template_mgr, "pitcher", pathfinder, object_y_lift=args.object_y_lift)
    all_object_ids["brick_0"] = place_object_at_navmesh(rigid_mgr, template_mgr, "foam", pathfinder, object_y_lift=args.object_y_lift)
    all_object_ids["brick_1"] = place_object_at_navmesh(rigid_mgr, template_mgr, "brick", pathfinder, object_y_lift=args.object_y_lift)

    frame_idx = 0
    frame_idx, phase_records, _ = capture_from_traj(
        sim=sim, agent=agent, poses=traj_poses[:800],
        out_dir=out_root, phase_name="before", frame_start=frame_idx,
        pathfinder=pathfinder, min_clearance=args.min_clearance,
        max_snap_dist=args.max_snap_dist,
    )
    event_log.extend(phase_records)

    # === Phase 2: DELETE + MOVE at frame 800, capture frames 800-1599 ===
    # Delete: bowl×2, mug×2 (desktop objects)
    for key in ["bowl_0", "bowl_1", "mug_0", "mug_1"]:
        oid = all_object_ids.pop(key, None)
        if oid is not None:
            rigid_mgr.remove_object_by_id(int(oid))
            event_log.append({"frame": 800, "type": "DELETE", "object_id": int(oid), "label": key.split("_")[0]})

    # Move: pitcher×2, brick×2 → new NavMesh points
    for key in ["pitcher_0", "pitcher_1", "brick_0", "brick_1"]:
        oid = all_object_ids.get(key)
        if oid is not None:
            obj = rigid_mgr.get_object_by_id(int(oid))
            if obj is not None:
                new_pos = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
                new_pos[1] += args.object_y_lift
                obj.translation = new_pos
                event_log.append({"frame": 800, "type": "MOVE", "object_id": int(oid), "label": key.split("_")[0],
                                  "new_position": [float(new_pos[0]), float(new_pos[1]), float(new_pos[2])]})

    frame_idx, phase_records, _ = capture_from_traj(
        sim=sim, agent=agent, poses=traj_poses[800:1600],
        out_dir=out_root, phase_name="mid", frame_start=frame_idx,
        pathfinder=pathfinder, min_clearance=args.min_clearance,
        max_snap_dist=args.max_snap_dist,
    )
    event_log.extend(phase_records)

    # === Phase 3: ADD at frame 1600, capture frames 1600-2399 ===
    add_labels = ["banana", "banana", "apple", "apple"]
    add_furniture = ["table_04", "table_04", "table_01", "table_01"]
    for label, fkey in zip(add_labels, add_furniture):
        oid = place_object_on_surface(rigid_mgr, template_mgr, label, furniture[fkey], 0.06, args.object_y_lift)
        if oid is not None:
            event_log.append({"frame": 1600, "type": "ADD", "object_id": int(oid), "label": label})

    frame_idx, phase_records, _ = capture_from_traj(
        sim=sim, agent=agent, poses=traj_poses[1600:2400],
        out_dir=out_root, phase_name="after", frame_start=frame_idx,
        pathfinder=pathfinder, min_clearance=args.min_clearance,
        max_snap_dist=args.max_snap_dist,
    )
    event_log.extend(phase_records)

    # === Write outputs ===
    with (out_root / "event_log.json").open("w", encoding="utf-8") as f:
        json.dump({
            "scene": args.scene_path,
            "total_frames": frame_idx,
            "events": [e for e in event_log if "type" in e],
        }, f, indent=2)

    import shutil
    shutil.copy(args.traj_file, out_root / "traj.txt")
    print(f"Saved {frame_idx} frames to {out_root}")
    sim.close()
```

- [ ] **Step 6: Verify syntax**

```bash
/home/ww/miniconda3/envs/habitat/bin/python -c "
compile(open('/home/ww/vv/habitat-data-collector/scripts/20260227/generate_dynamic_benchmark.py').read(), 'gen.py', 'exec')
print('Syntax OK')
"
```
Expected: `Syntax OK`

- [ ] **Step 7: Smoke test — 30 frames**

```bash
export LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib:$LD_LIBRARY_PATH
/home/ww/miniconda3/envs/habitat/bin/python \
  /home/ww/vv/habitat-data-collector/scripts/20260227/generate_dynamic_benchmark.py \
  --scene-path /home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json \
  --scene-dataset-config /home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json \
  --ycb-dir /home/ww/vv/habitat_data/objects/ycb \
  --traj-file /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
  --output /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v4/smoke \
  --width 640 --height 480 --seed 7 --max-snap-dist 0.5
```
Then verify: `ls outputs/dynamic_benchmark_v4/smoke/rgb/ | wc -l` → expect ~30 frames and event_log.json with events.

- [ ] **Step 8: Commit**

```bash
git -C /home/ww/vv/habitat-data-collector add scripts/20260227/generate_dynamic_benchmark.py
git -C /home/ww/vv/habitat-data-collector commit -m "feat: add multi-object multi-phase benchmark generator"
```

---

### Task 2: Generate full 2400-frame dataset

**Files:**
- Creates: `outputs/dynamic_benchmark_v4/full/`

- [ ] **Step 1: Run full generation**

```bash
export LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib:$LD_LIBRARY_PATH
/home/ww/miniconda3/envs/habitat/bin/python \
  /home/ww/vv/habitat-data-collector/scripts/20260227/generate_dynamic_benchmark.py \
  --scene-path /home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json \
  --scene-dataset-config /home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json \
  --ycb-dir /home/ww/vv/habitat_data/objects/ycb \
  --traj-file /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
  --output /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v4/full \
  --width 640 --height 480 --seed 7 --max-snap-dist 0.5
```

This runs ~60-90 minutes. Verify output count and event_log.

- [ ] **Step 2: Verify dataset**

```bash
echo "RGB: $(ls outputs/dynamic_benchmark_v4/full/rgb/ | wc -l)"
echo "Depth: $(ls outputs/dynamic_benchmark_v4/full/depth/ | wc -l)"
test -f outputs/dynamic_benchmark_v4/full/event_log.json && echo "event_log OK"
python3 -c "import json; e=json.load(open('outputs/dynamic_benchmark_v4/full/event_log.json')); 
  print(f'Events: {len(e[\"events\"])}')"
```
Expected: events contain DELETE/MOVE/ADD at frames 800/1600.

---

### Task 3: Run pipeline on generated dataset

- [ ] **Step 1: Run pipeline**

```bash
cd /home/ww/tmp/oviovo-pr && python3 run_room0_full_eval.py \
  --dataset-root outputs/dynamic_benchmark_v4/full \
  --num-frames 0 \
  --config-path configs/apt0_dynamic_benchmark.yaml \
  --proposal-backend sam2 --proposal-device cuda \
  --sam-repo-root /home/ww/vv/paper2/DovSG/third_party/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/paper2/DovSG/checkpoints/segment-anything-2 \
  --sam-encoder hiera_l \
  --experiment-name dynamic_bench_v4
```

~2 hours. Verify frame_metrics has new fields.

---

### Task 4: Adapt eval script for V4 and run evaluation

- [ ] **Step 1: Add V4 eval path to scripts/eval_dynamic_benchmark.py**

```python
def evaluate_v4(event_log: dict, metrics: list[dict]) -> dict:
    events = event_log.get("events", [])
    delete_frame = next((e["frame"] for e in events if e["type"] == "DELETE"), None)
    move_frame = next((e["frame"] for e in events if e["type"] == "MOVE"), None)
    add_frame = next((e["frame"] for e in events if e["type"] == "ADD"), None)

    ghost_total = 0; moved_total = 0; new_total = 0
    fp_ghost = 0
    for m in metrics:
        fid = int(m.get("frame_id", 0))
        ghost = m.get("ghost_object_count", 0)
        moved = len(m.get("moved_this_frame", []))
        new_cand = len(m.get("new_candidate_this_frame", []))

        if ghost > 0:
            if delete_frame and delete_frame + 50 <= fid <= delete_frame + 200:
                ghost_total += 1
            elif fid < 800:
                fp_ghost += 1
        if moved > 0 and move_frame and move_frame + 50 <= fid <= move_frame + 200:
            moved_total += moved
        if new_cand > 0 and add_frame and add_frame + 50 <= fid <= add_frame + 200:
            new_total += new_cand

    return {
        "ghost_detected_in_window": ghost_total,
        "moved_detected_in_window": moved_total >= 1,
        "new_object_detected_in_window": new_total >= 1,
        "false_positive_ghost": fp_ghost,
    }
```

- [ ] **Step 2: Run eval**

```bash
python3 scripts/eval_dynamic_benchmark.py \
  --mode v4 \
  --event-log outputs/dynamic_benchmark_v4/full/event_log.json \
  --frame-metrics outputs/dynamic_bench_v4/room0/frame_metrics.jsonl \
  --output outputs/dynamic_benchmark_v4/benchmark_report.json
python3 -c "import json; print(json.dumps(json.load(open('outputs/dynamic_benchmark_v4/benchmark_report.json')), indent=2))"
```

- [ ] **Step 3: Commit results**

```bash
git add scripts/eval_dynamic_benchmark.py outputs/dynamic_benchmark_v4/
git commit -m "feat: V4 dynamic benchmark — eval results"
```

---