# Dynamic Benchmark V3 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用两段式静态场景拼接生成 apt_0 2400 帧动态 benchmark 数据集，跑 pipeline 评估 TSDF 驱动三级判决。

**Architecture:** 修改已有轨迹重放脚本支持 YCB 物体配置，两次独立 Habitat 启动渲染"变化前/后"各 1200 帧，合并为 2400 帧连续序列，跑 pipeline 产出 frame_metrics.jsonl，用评估脚本对比 event_log.json。

**Tech Stack:** Python 3.9 (conda env habitat), Habitat Sim 0.3.3, NumPy, imageio, PyYAML

## Global Constraints

- OVIOVO pipeline 核心逻辑不改
- TSDF maintenance 判定逻辑不改
- ReplicaCAD 路径：`/home/ww/vv/habitat_data/scene_datasets/replica_cad`
- YCB 路径：`/home/ww/vv/habitat_data/objects/ycb`
- 轨迹文件：`/home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt`
- Habitat conda env：`/home/ww/miniconda3/envs/habitat/bin/python`
- NVIDIA EGL：`LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib`
- 所有输出放在 `outputs/dynamic_benchmark_v3/`
- 过渡帧 = 1200（10 的整数倍），检测窗口 = [1250, 1400]

---

### Task 1: Adapt replay script for apt_0 + YCB

**Files:**
- Create: `/home/ww/vv/habitat-data-collector/scripts/20260227/replay_apt0_traj.py`
- Consumes: `/home/ww/vv/habitat-data-collector/scripts/20260227/replay_room0_traj_with_multi_objects.py` (base)

**Interfaces:**
- Produces: `replay_apt0_traj.py` with args `--traj-file`, `--scene-path`, `--scene-dataset-config`, `--objects-dir`, `--object-plan`, `--output`, `--frames`, `--width`, `--height`, `--depth-scale`, `--seed`

- [ ] **Step 1: Copy base script and verify it exists**

```bash
cp /home/ww/vv/habitat-data-collector/scripts/20260227/replay_room0_traj_with_multi_objects.py \
   /home/ww/vv/habitat-data-collector/scripts/20260227/replay_apt0_traj.py
echo "Copied"
```

- [ ] **Step 2: Replace args and remove room0/align logic**

Replace the argument parser block (lines 188-198). Change `--room-dir` to `--traj-file`, add `--objects-dir` and `--object-plan`, add `--frames`, add `--seed`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Replay apt_0 traj with YCB objects")
    parser.add_argument("--traj-file", required=True, help="Path to traj.txt (16-value 4x4 matrices)")
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--scene-dataset-config", required=True)
    parser.add_argument("--objects-dir", required=True, help="Directory containing YCB object configs")
    parser.add_argument("--object-plan", required=True, help="JSON file listing object handles to place")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-scale", type=float, default=6553.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
```

Remove the old room_dir/candidates_file/align logic (lines 200-251). Replace with:

```python
    out_root = Path(args.output)
    rgb_dir = out_root / "rgb"
    depth_dir = out_root / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    traj_file = Path(args.traj_file)
    poses = load_traj_poses(traj_file)
    if not poses:
        raise RuntimeError("No valid poses found in traj.txt")

    with open(args.object_plan, "r", encoding="utf-8") as f:
        object_plan = json.load(f)
    object_handles = object_plan.get("handles", [])
```

- [ ] **Step 3: Replace hardcoded object loading with YCB**

Replace lines 253-278 (template loading + object placement) with:

```python
    sim = build_sim(args.scene_path, args.scene_dataset_config, args.width, args.height)
    agent = sim.initialize_agent(0)

    template_mgr = sim.get_object_template_manager()
    # Load all YCB object configs
    objects_dir = Path(args.objects_dir)
    for config_file in sorted(objects_dir.glob("*.object_config.json")):
        template_mgr.load_configs(str(config_file))

    rigid_mgr = sim.get_rigid_object_manager()

    # Place objects in front of the first camera pose
    first_pose_pos = poses[0][0]
    first_pose_quat = poses[0][1]
    cam_rot = quat_xyzw_to_rotmat(first_pose_quat)
    forward = cam_rot @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    placed = []
    distance = 1.5  # meters in front of camera
    spacing = 0.5   # lateral spacing
    for i, handle_keyword in enumerate(object_handles):
        # Find matching YCB template
        matching = template_mgr.get_file_template_handles(handle_keyword)
        if not matching:
            print(f"Warning: no YCB object matching '{handle_keyword}'")
            continue
        handle = matching[0]

        obj = rigid_mgr.add_object_by_template_handle(handle)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # Place in front of camera, offset laterally
        right = cam_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        offset = (i % 4) - 1.5  # spread across camera view
        p = first_pose_pos + forward * distance + right * offset * spacing
        p[1] += 0.05  # slight lift
        obj.translation = p.astype(np.float32)

        placed.append({
            "handle_keyword": handle_keyword,
            "resolved_template": handle,
            "position": [float(p[0]), float(p[1]), float(p[2])],
            "object_id": int(obj.object_id),
        })
```

- [ ] **Step 4: Keep rendering loop unchanged, add frame limit**

The rendering loop (lines 280-300) stays the same. Just add a frame limit before it:

```python
    num_frames = min(args.frames, len(poses))
    for idx in range(num_frames):
        cam_pos, quat_xyzw = poses[idx]
        # ... existing rendering code ...
```

Also add event_log export at the end:

```python
    event_log = {
        "scene": args.scene_path,
        "num_frames": num_frames,
        "objects_placed": placed,
    }
    with (out_root / "event_log.json").open("w", encoding="utf-8") as f:
        json.dump(event_log, f, indent=2)
    print(f"Saved {num_frames} frames to {out_root}")
```

- [ ] **Step 5: Verify syntax**

```bash
/home/ww/miniconda3/envs/habitat/bin/python -c "
compile(open('/home/ww/vv/habitat-data-collector/scripts/20260227/replay_apt0_traj.py').read(), 'replay_apt0_traj.py', 'exec')
print('Syntax OK')
"
```
Expected: `Syntax OK`

- [ ] **Step 6: Smoke test with 5 frames, 1 object**

```bash
mkdir -p /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v3/smoke_test
echo '{"handles": ["sphere"]}' > /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v3/smoke_plan.json

export LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib:$LD_LIBRARY_PATH
/home/ww/miniconda3/envs/habitat/bin/python \
  /home/ww/vv/habitat-data-collector/scripts/20260227/replay_apt0_traj.py \
  --traj-file /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
  --scene-path /home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json \
  --scene-dataset-config /home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json \
  --objects-dir /home/ww/vv/habitat_data/objects/ycb \
  --object-plan /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v3/smoke_plan.json \
  --output /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v3/smoke_test \
  --frames 5 --width 640 --height 480 --seed 7
```
Expected: 5 frames generated, rgb/ and depth/ each have 5 files.

- [ ] **Step 7: Commit**

```bash
git -C /home/ww/vv/habitat-data-collector add scripts/20260227/replay_apt0_traj.py
git -C /home/ww/vv/habitat-data-collector commit -m "feat: add replay_apt0_traj.py — traj replay with YCB objects"
```

---

### Task 2: Create object plans for Run A and Run B

**Files:**
- Create: `outputs/dynamic_benchmark_v3/run_a_objects.json`
- Create: `outputs/dynamic_benchmark_v3/run_b_objects.json`

**Interfaces:**
- Produces: Two JSON files consumable by `replay_apt0_traj.py --object-plan`

- [ ] **Step 1: Create run_a_objects.json**

Create `/home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v3/run_a_objects.json`:

```json
{
  "description": "Run A: apt_0 + 8 YCB objects (4 to DELETE, 4 to MOVE at old positions)",
  "handles": [
    "bowl",
    "bowl",
    "cup",
    "cup",
    "chair",
    "chair",
    "donut",
    "donut"
  ]
}
```

- [ ] **Step 2: Create run_b_objects.json**

Create `/home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark_v3/run_b_objects.json`:

```json
{
  "description": "Run B: apt_0 + 8 YCB objects (4 MOVED, 4 ADDED)",
  "handles": [
    "chair",
    "chair",
    "donut",
    "donut",
    "sphere",
    "sphere",
    "apple",
    "apple"
  ]
}
```

- [ ] **Step 3: Verify JSON validity**

```bash
python3 -c "import json; json.load(open('outputs/dynamic_benchmark_v3/run_a_objects.json')); json.load(open('outputs/dynamic_benchmark_v3/run_b_objects.json')); print('Both valid')"
```
Expected: `Both valid`

- [ ] **Step 4: Check that all handles exist in YCB**

```bash
/home/ww/miniconda3/envs/habitat/bin/python -c "
import habitat_sim
import os, json
from pathlib import Path

ycb = Path('/home/ww/vv/habitat_data/objects/ycb')
# Initialize sim just to get template manager
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = '/home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json'
sim_cfg.scene_dataset_config_file = '/home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json'
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, []))
tm = sim.get_object_template_manager()
for f in sorted(ycb.glob('*.object_config.json')):
    tm.load_configs(str(f))

for plan_file in ['outputs/dynamic_benchmark_v3/run_a_objects.json', 'outputs/dynamic_benchmark_v3/run_b_objects.json']:
    with open(plan_file) as f:
        plan = json.load(f)
    for h in plan['handles']:
        matches = tm.get_file_template_handles(h)
        print(f'{plan_file.split(\"/\")[-1]}: \"{h}\" -> {len(matches)} matches')
sim.close()
" 2>&1 | grep -v Warning | grep -v Error
```
Expected: Each handle has ≥1 match.

- [ ] **Step 5: Commit**

```bash
git add outputs/dynamic_benchmark_v3/run_a_objects.json outputs/dynamic_benchmark_v3/run_b_objects.json
git commit -m "feat: add YCB object plans for V3 benchmark (Run A + Run B)"
```

---

### Task 3: Render Run A (1200 frames)

**Files:**
- Creates: `outputs/dynamic_benchmark_v3/run_a/` (rgb/, depth/, event_log.json)

- [ ] **Step 1: Run Run A**

```bash
export LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib:$LD_LIBRARY_PATH
/home/ww/miniconda3/envs/habitat/bin/python \
  /home/ww/vv/habitat-data-collector/scripts/20260227/replay_apt0_traj.py \
  --traj-file /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
  --scene-path /home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json \
  --scene-dataset-config /home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json \
  --objects-dir /home/ww/vv/habitat_data/objects/ycb \
  --object-plan outputs/dynamic_benchmark_v3/run_a_objects.json \
  --output outputs/dynamic_benchmark_v3/run_a \
  --frames 1200 --width 640 --height 480 --seed 7
```

This takes ~30-60 min. Verify output: `ls outputs/dynamic_benchmark_v3/run_a/rgb/ | wc -l` → 1200

- [ ] **Step 2: Verify Run A output**

```bash
test $(ls outputs/dynamic_benchmark_v3/run_a/rgb/ | wc -l) -eq 1200 && echo "RGB OK"
test $(ls outputs/dynamic_benchmark_v3/run_a/depth/ | wc -l) -eq 1200 && echo "Depth OK"
test -f outputs/dynamic_benchmark_v3/run_a/event_log.json && echo "Event log OK"
```
Expected: All three OK.

---

### Task 4: Render Run B (1200 frames)

Same as Task 3 but with `run_b_objects.json` → `run_b/`.

- [ ] **Step 1: Run Run B**

```bash
export LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib:$LD_LIBRARY_PATH
/home/ww/miniconda3/envs/habitat/bin/python \
  /home/ww/vv/habitat-data-collector/scripts/20260227/replay_apt0_traj.py \
  --traj-file /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
  --scene-path /home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json \
  --scene-dataset-config /home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json \
  --objects-dir /home/ww/vv/habitat_data/objects/ycb \
  --object-plan outputs/dynamic_benchmark_v3/run_b_objects.json \
  --output outputs/dynamic_benchmark_v3/run_b \
  --frames 1200 --width 640 --height 480 --seed 7
```

- [ ] **Step 2: Verify Run B output**

```bash
test $(ls outputs/dynamic_benchmark_v3/run_b/rgb/ | wc -l) -eq 1200 && echo "RGB OK"
test $(ls outputs/dynamic_benchmark_v3/run_b/depth/ | wc -l) -eq 1200 && echo "Depth OK"
```

---

### Task 5: Merge Run A + Run B into 2400-frame dataset

**Files:**
- Creates: `outputs/dynamic_benchmark_v3/merged/` + `event_log.json` + `traj.txt`

- [ ] **Step 1: Merge with frame renumbering**

```bash
mkdir -p outputs/dynamic_benchmark_v3/merged/rgb outputs/dynamic_benchmark_v3/merged/depth

# Run A: frames 0-1199 (keep original numbering)
cp outputs/dynamic_benchmark_v3/run_a/rgb/frame*.jpg outputs/dynamic_benchmark_v3/merged/rgb/
cp outputs/dynamic_benchmark_v3/run_a/depth/depth*.png outputs/dynamic_benchmark_v3/merged/depth/

# Run B: frames 0-1199 → renumber to 1200-2399
python3 -c "
import os, shutil
for i in range(1200):
    old_rgb = f'outputs/dynamic_benchmark_v3/run_b/rgb/frame{i:06d}.jpg'
    new_rgb = f'outputs/dynamic_benchmark_v3/merged/rgb/frame{i+1200:06d}.jpg'
    old_depth = f'outputs/dynamic_benchmark_v3/run_b/depth/depth{i:06d}.png'
    new_depth = f'outputs/dynamic_benchmark_v3/merged/depth/depth{i+1200:06d}.png'
    if os.path.exists(old_rgb):
        shutil.copy2(old_rgb, new_rgb)
    if os.path.exists(old_depth):
        shutil.copy2(old_depth, new_depth)
print('Merge done')
"

test $(ls outputs/dynamic_benchmark_v3/merged/rgb/ | wc -l) -eq 2400 && echo "2400 RGB OK"
test $(ls outputs/dynamic_benchmark_v3/merged/depth/ | wc -l) -eq 2400 && echo "2400 Depth OK"
```

- [ ] **Step 2: Copy trajectory and create event_log.json**

```bash
# Copy the 2400-frame traj
cp /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
   outputs/dynamic_benchmark_v3/merged/traj.txt

# Create GT event_log
python3 -c "
import json
event_log = {
    'scene': 'apt_0',
    'total_frames': 2400,
    'transition_frame': 1200,
    'detection_window': [1250, 1400],
    'expected': {
        'disappeared': {'count': 4, 'labels': ['bowl', 'bowl', 'cup', 'cup']},
        'moved': {'count': 4, 'labels': ['chair', 'chair', 'donut', 'donut']},
        'new_object': {'count': 4, 'labels': ['sphere', 'sphere', 'apple', 'apple']}
    }
}
with open('outputs/dynamic_benchmark_v3/merged/event_log.json', 'w') as f:
    json.dump(event_log, f, indent=2)
print('event_log.json written')
"
```

- [ ] **Step 3: Convert traj format for ReplicaRoom0Dataset**

```bash
python3 -c "
import numpy as np
# The editable dataset's traj is already 16-value 4x4 format — just copy
# But also ensure it can be loaded by our dataset class
import sys; sys.path.insert(0, '.')
from src.datasets.replica import ReplicaRoom0Dataset
ds = ReplicaRoom0Dataset('outputs/dynamic_benchmark_v3/merged')
print(f'Dataset frames: {len(ds)}')
" 2>&1 | grep -v Warning
```
Expected: `Dataset frames: 2400`

- [ ] **Step 4: Also rename PNG→JPG if needed**

```bash
# Replay script outputs frame*.jpg already, no conversion needed
ls outputs/dynamic_benchmark_v3/merged/rgb/ | head -3
```
Expected: `frame000000.jpg` etc.

- [ ] **Step 5: Commit**

```bash
git add outputs/dynamic_benchmark_v3/merged/event_log.json outputs/dynamic_benchmark_v3/merged/traj.txt
git commit -m "feat: merge Run A + Run B into 2400f dataset with GT event_log"
```

---

### Task 6: Run pipeline on merged dataset

**Files:**
- Creates: pipeline output in `outputs/dynamic_benchmark_v3_final/`

- [ ] **Step 1: Run pipeline**

```bash
cd /home/ww/tmp/oviovo-pr && python3 run_room0_full_eval.py \
  --dataset-root outputs/dynamic_benchmark_v3/merged \
  --num-frames 0 \
  --config-path configs/apt0_dynamic_benchmark.yaml \
  --proposal-backend sam2 --proposal-device cuda \
  --sam-repo-root /home/ww/vv/paper2/DovSG/third_party/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/paper2/DovSG/checkpoints/segment-anything-2 \
  --sam-encoder hiera_l \
  --experiment-name dynamic_benchmark_v3_final
```

This takes ~2 hours for 2400 frames. Run in background if needed.

- [ ] **Step 2: Verify frame_metrics has new fields**

```bash
python3 -c "
import json
with open('outputs/dynamic_benchmark_v3_final/room0/frame_metrics.jsonl') as f:
    d = json.loads(f.readline())
    print('ghost_object_count' in d, 'moved_this_frame' in d, 'new_candidate_this_frame' in d)
"
```
Expected: `True True True`

---

### Task 7: Adapt eval script and run evaluation

**Files:**
- Modify: `scripts/eval_dynamic_benchmark.py`

- [ ] **Step 1: Adapt eval script for V3 format**

The V3 event_log has `transition_frame`, `detection_window`, and `expected` dict. Add a new evaluation path:

```python
def evaluate_v3(event_log: dict, metrics: list[dict]) -> dict:
    """Evaluate V3 two-pass benchmark."""
    transition = event_log["transition_frame"]
    window = event_log["detection_window"]  # [1250, 1400]
    expected = event_log["expected"]

    ghost_in_window = 0
    moved_in_window = 0
    new_in_window = 0
    false_positive_ghost = 0

    for m in metrics:
        f = int(m.get("frame_id", 0))
        ghost = m.get("ghost_object_count", 0)
        moved = len(m.get("moved_this_frame", []))
        new_cand = len(m.get("new_candidate_this_frame", []))

        if window[0] <= f <= window[1]:
            if ghost > 0:
                ghost_in_window = 1
            if moved > 0:
                moved_in_window += moved
            if new_cand > 0:
                new_in_window += new_cand
        elif f < transition and ghost > 0:
            false_positive_ghost += 1  # ghost before transition = false positive

    return {
        "disappeared_detected": ghost_in_window > 0,
        "moved_detected": moved_in_window >= 1,
        "new_object_detected": new_in_window >= 1,
        "false_positive_ghost_frames": false_positive_ghost,
        "transition_frame": transition,
        "detection_window": window,
    }
```

- [ ] **Step 2: Add --mode v3 flag to CLI**

```python
parser.add_argument("--mode", choices=["v1", "v3"], default="v1")
```

In `main()`, when `args.mode == "v3"`, load the entire event_log dict (not list) and call `evaluate_v3`.

- [ ] **Step 3: Run eval**

```bash
python3 scripts/eval_dynamic_benchmark.py \
  --mode v3 \
  --event-log outputs/dynamic_benchmark_v3/merged/event_log.json \
  --frame-metrics outputs/dynamic_benchmark_v3_final/room0/frame_metrics.jsonl \
  --output outputs/dynamic_benchmark_v3/benchmark_report.json
```

- [ ] **Step 4: Verify report**

```bash
python3 -c "
import json
with open('outputs/dynamic_benchmark_v3/benchmark_report.json') as f:
    r = json.load(f)
print(f'DISAPPEARED detected: {r[\"disappeared_detected\"]}')
print(f'MOVED detected: {r[\"moved_detected\"]}')
print(f'NEW OBJECT detected: {r[\"new_object_detected\"]}')
print(f'False positive ghost frames: {r[\"false_positive_ghost_frames\"]}')
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_dynamic_benchmark.py
git commit -m "feat: add V3 eval mode for two-pass benchmark evaluation"
```

---
