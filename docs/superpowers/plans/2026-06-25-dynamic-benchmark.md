# Dynamic Benchmark 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 apt_0 12 变更事件动态 benchmark：用 Habitat 生成带 GT 标注的 RGB-D 数据集，跑 pipeline 获取 frame 级指标，对比 event_log 产出评估报告。

**Architecture:** 两个独立阶段。Stage 1 用 Habitat Sim 生成带精确帧号 GT 的 RGB-D 数据集。Stage 2 跑 OVIOVO pipeline（不改核心逻辑，仅 frame_metrics 加 3 字段），用独立评估脚本对比 event_log.json 和 frame_metrics.jsonl 产出报告。

**Tech Stack:** Python 3.9 (conda env habitat), Habitat Sim 0.3.3, NumPy, imageio, PyYAML, pytest

## File Structure

```
outputs/dynamic_benchmark/          # All benchmark artifacts
├── event_plan.json                 # Task 1: 12 events with frame numbers
├── dataset/                        # Task 3: Habitat-generated RGB-D
│   ├── rgb/frame_000000.jpg ... 
│   ├── depth/depth_000000.png ...
│   ├── traj.txt
│   ├── changes.json
│   └── event_log.json             # GT with detection windows
├── pipeline_output/                # Task 6: OVIOVO pipeline run
│   └── frame_metrics.jsonl
└── benchmark_report.json           # Task 6: eval script output

scripts/eval_dynamic_benchmark.py   # Task 5: standalone eval script
```

## Global Constraints

- OVIOVO pipeline 核心逻辑不改（TSDF maintenance、object_update、proposal 等模块不变）
- Habitat-data-collector 架构不改（generator 已有 --event-plan，只改 event_log 导出格式）
- 评估脚本独立于 pipeline，读 JSON 文件做对比，不修改 pipeline 代码
- 所有输出文件放在 `outputs/dynamic_benchmark/` 下
- Habitat conda env 路径：`/home/ww/miniconda3/envs/habitat/bin/python`
- ReplicaCAD 路径：`/home/ww/vv/habitat_data/scene_datasets/replica_cad`
- YCB 路径：`/home/ww/vv/habitat_data/objects/ycb`

---

### Task 1: Create event_plan.json for apt_0 12 events

**Files:**
- Create: `outputs/dynamic_benchmark/event_plan.json`

**Interfaces:**
- Produces: JSON file consumable by `replica_dynamic_rgbd_generator.py --event-plan`

- [ ] **Step 1: Write the event plan**

Create `outputs/dynamic_benchmark/event_plan.json`:

```json
{
  "scene": "apt_0",
  "total_frames": 2400,
  "initial_objects": [],
  "events": [
    {"frame": 150,  "type": "remove", "handle_keyword": "lamp"},
    {"frame": 300,  "type": "remove", "handle_keyword": "lamp"},
    {"frame": 450,  "type": "remove", "handle_keyword": "chair"},
    {"frame": 600,  "type": "remove", "handle_keyword": "chair"},
    {"frame": 750,  "type": "move",   "handle_keyword": "chair", "delta": [4.0, 0.6, -4.5]},
    {"frame": 900,  "type": "move",   "handle_keyword": "table", "delta": [1.0, 0.0, -2.5]},
    {"frame": 1050, "type": "move",   "handle_keyword": "chair", "delta": [4.6, -0.2, 3.0]},
    {"frame": 1200, "type": "move",   "handle_keyword": "table", "delta": [-5.8, 0.0, -5.2]},
    {"frame": 1350, "type": "add",    "handle_keyword": "bowl"},
    {"frame": 1500, "type": "add",    "handle_keyword": "bowl"},
    {"frame": 1650, "type": "add",    "handle_keyword": "cup"},
    {"frame": 1800, "type": "add",    "handle_keyword": "cup"}
  ]
}
```

- [ ] **Step 2: Verify JSON is valid**

Run:
```bash
python3 -c "import json; json.load(open('outputs/dynamic_benchmark/event_plan.json')); print('Valid')"
```
Expected: `Valid`

- [ ] **Step 3: Commit**

```bash
git add outputs/dynamic_benchmark/event_plan.json
git commit -m "feat: add apt_0 12-event benchmark plan"
```

---

### Task 2: Add event_log.json with GT annotations to generator

**Files:**
- Modify: `/home/ww/vv/habitat-data-collector/scripts/20260227/replica_dynamic_rgbd_generator.py`
  - Add GT annotation fields to the event_log entries
  - Write `event_log.json` at end of generation

**Interfaces:**
- Consumes: event plan JSON (existing format)
- Produces: `event_log.json` in output dir with `expected_judgment` and `detection_window` fields

- [ ] **Step 1: Read the generator's event processing section**

The generator already logs events at lines ~870-1050. Each event entry has `frame`, `type`, `handle_keyword`, etc. We need to add two fields to each entry in the `event_log` list.

Find the section where `event_log.append(...)` is called for remove/move/add events (around lines 870-1050). Add two keys to each append dict:

```python
# For "remove" events:
info["expected_judgment"] = "DISAPPEARED"
info["detection_window"] = [frame + 50, frame + 200]

# For "move" events:
info["expected_judgment"] = "MOVED"
info["detection_window"] = [frame + 50, frame + 200]

# For "add" events:
info["expected_judgment"] = "NEW_OBJECT"
info["detection_window"] = [frame + 50, frame + 200]
```

- [ ] **Step 2: Write event_log.json at end of main()**

At the end of `main()`, after the frame loop, add:

```python
    # Write GT-annotated event log
    event_log_path = output / "event_log.json"
    with open(event_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "scene": args.scene_path,
            "total_frames": args.frames,
            "events": event_log,
        }, f, indent=2)
    print(f"Event log written to {event_log_path}")
```

- [ ] **Step 3: Test import**

Run:
```bash
/home/ww/miniconda3/envs/habitat/bin/python -c "
import sys; sys.path.insert(0, '/home/ww/vv/habitat-data-collector/scripts/20260227')
# Just verify the script parses without syntax errors
compile(open('/home/ww/vv/habitat-data-collector/scripts/20260227/replica_dynamic_rgbd_generator.py').read(), 'generator.py', 'exec')
print('Syntax OK')
"
```
Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
# This is outside the PR repo — commit in habitat-data-collector
git -C /home/ww/vv/habitat-data-collector add scripts/20260227/replica_dynamic_rgbd_generator.py
git -C /home/ww/vv/habitat-data-collector commit -m "feat: add GT annotations to event_log export"
```

---

### Task 3: Generate apt_0 dynamic dataset (run Stage 1)

**Files:**
- Creates: `outputs/dynamic_benchmark/dataset/` (rgb/, depth/, traj.txt, changes.json, event_log.json)

**Interfaces:**
- Consumes: event_plan.json, ReplicaCAD apt_0, YCB objects
- Produces: 2400-frame RGB-D dataset with GT event log

- [ ] **Step 1: Copy trajectory and create output dir**

```bash
mkdir -p /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark/dataset
cp /home/ww/vv/dataset/Replica/room0/traj.txt /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark/dataset/traj.txt
```

- [ ] **Step 2: Run Habitat generator (smoke test: 30 frames)**

```bash
/home/ww/miniconda3/envs/habitat/bin/python \
  /home/ww/vv/habitat-data-collector/scripts/20260227/replica_dynamic_rgbd_generator.py \
  --scene-path /home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json \
  --scene-dataset-config /home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json \
  --objects-dir /home/ww/vv/habitat_data/objects/ycb \
  --event-plan /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark/event_plan.json \
  --output /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark/dataset \
  --frames 30 --width 640 --height 480 --seed 7 \
  --camera-policy safe_sample --sensor-height 1.5
```
Expected: 30 frames generated, no errors. rgb/ and depth/ dirs contain 30 files each.

- [ ] **Step 3: Verify output structure**

```bash
ls outputs/dynamic_benchmark/dataset/rgb/ | wc -l  # Expected: 30
ls outputs/dynamic_benchmark/dataset/depth/ | wc -l  # Expected: 30
test -f outputs/dynamic_benchmark/dataset/event_log.json && echo "event_log OK"
```
Expected: 30, 30, event_log OK

- [ ] **Step 4: Run full 2400-frame generation (background)**

```bash
/home/ww/miniconda3/envs/habitat/bin/python \
  /home/ww/vv/habitat-data-collector/scripts/20260227/replica_dynamic_rgbd_generator.py \
  --scene-path /home/ww/vv/habitat_data/scene_datasets/replica_cad/configs/scenes/apt_0.scene_instance.json \
  --scene-dataset-config /home/ww/vv/habitat_data/scene_datasets/replica_cad/replicaCAD.scene_dataset_config.json \
  --objects-dir /home/ww/vv/habitat_data/objects/ycb \
  --event-plan /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark/event_plan.json \
  --output /home/ww/tmp/oviovo-pr/outputs/dynamic_benchmark/dataset \
  --frames 2400 --width 640 --height 480 --seed 7 \
  --camera-policy safe_sample --sensor-height 1.5
```

This will run for several hours. Verify event_log.json exists after completion.

- [ ] **Step 5: Commit**

```bash
git add outputs/dynamic_benchmark/dataset/event_log.json outputs/dynamic_benchmark/dataset/traj.txt
git commit -m "feat: add generated dynamic benchmark dataset GT"
```

---

### Task 4: Add ghost/moved/new_candidate fields to frame_metrics

**Files:**
- Modify: `src/pipelines/main_pipeline.py` — the `last_frame_debug` dict

**Interfaces:**
- Produces: `frame_metrics.jsonl` now includes `ghost_object_count`, `moved_this_frame`, `new_candidate_this_frame`

- [ ] **Step 1: Add three fields to last_frame_debug**

In `src/pipelines/main_pipeline.py`, find the `self.last_frame_debug = {` block. Add these three entries before the closing `}`:

```python
            "ghost_object_count": int(
                sum(1 for o in self.state.objects.values()
                    if o.state == ObjectState.GHOST)
            ),
            "moved_this_frame": list(
                getattr(self.dynamic_maintenance, "last_moved_object_ids", [])
            ),
            "new_candidate_this_frame": list(
                getattr(self.dynamic_maintenance, "last_new_candidate_ids", [])
            ),
```

- [ ] **Step 2: Add tracking attributes to DynamicMaintenanceModule**

In `src/modules/dynamic_maintenance.py`, add to `__init__`:

```python
        self.last_moved_object_ids: List[int] = []
        self.last_new_candidate_ids: List[int] = []
```

In `process()`, at the MOVED detection point, append the object_id:

```python
            if self._check_moved(obj, volume, config):
                ...
                self.last_moved_object_ids.append(int(obj.object_id))
```

Reset at the beginning of each `process()` call:

```python
        self.last_moved_object_ids.clear()
        self.last_new_candidate_ids.clear()
```

For `last_new_candidate_ids`, append the provisional_id after creating a ProvisionalObject in the NEW OBJECT section.

- [ ] **Step 3: Verify existing tests still pass**

```bash
python3 -m pytest tests/test_dynamic_maintenance.py tests/test_dynamic_maintenance_integration.py -v 2>&1 | tail -5
```
Expected: 22 passed

- [ ] **Step 4: Commit**

```bash
git add src/pipelines/main_pipeline.py src/modules/dynamic_maintenance.py
git commit -m "feat: add ghost/moved/new_candidate tracking to frame_metrics"
```

---

### Task 5: Create eval_dynamic_benchmark.py

**Files:**
- Create: `scripts/eval_dynamic_benchmark.py`

**Interfaces:**
- Consumes: `event_log.json` + `frame_metrics.jsonl`
- Produces: `benchmark_report.json` + terminal output

- [ ] **Step 1: Write the eval script**

Create `scripts/eval_dynamic_benchmark.py`:

```python
#!/usr/bin/env python3
"""Compare dynamic benchmark event_log against pipeline frame_metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_event_log(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("events", [])


def load_frame_metrics(path: Path) -> list[dict]:
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def evaluate(event_log: list[dict], metrics: list[dict]) -> dict:
    results = []
    ghost_baseline = 0
    hits = 0
    total = len(event_log)
    total_delay = 0

    for event in event_log:
        event_frame = int(event["frame"])
        expected = event.get("expected_judgment", "")
        window = event.get("detection_window", [event_frame + 50, event_frame + 200])
        ev_type = event.get("type", "")

        detected = False
        detect_frame = None

        for m in metrics:
            f = int(m.get("frame_id", 0))
            if f < window[0]:
                ghost_baseline = m.get("ghost_object_count", 0)
                continue
            if f > window[1]:
                break

            if ev_type == "remove":
                ghost_now = m.get("ghost_object_count", 0)
                if ghost_now > ghost_baseline:
                    detected = True
                    detect_frame = f
                    break
            elif ev_type == "move":
                moved_ids = m.get("moved_this_frame", [])
                if len(moved_ids) > 0:
                    detected = True
                    detect_frame = f
                    break
            elif ev_type == "add":
                new_ids = m.get("new_candidate_this_frame", [])
                if len(new_ids) > 0:
                    detected = True
                    detect_frame = f
                    break

        delay = (detect_frame - event_frame) if detected else None
        if detected:
            hits += 1
            if delay is not None:
                total_delay += delay

        results.append({
            "event": event,
            "detected": detected,
            "detect_frame": detect_frame,
            "delay_frames": delay,
        })

    # False positive check: count frames with ghost > 0 outside any detection window
    all_windows = [(e["detection_window"][0], e["detection_window"][1]) for e in event_log]
    false_positives = 0
    for m in metrics:
        f = int(m.get("frame_id", 0))
        ghost = m.get("ghost_object_count", 0)
        if ghost > 0:
            in_any_window = any(lo <= f <= hi for lo, hi in all_windows)
            if not in_any_window:
                false_positives += 1

    return {
        "total_events": total,
        "detected": hits,
        "missed": total - hits,
        "recall": hits / total if total > 0 else 0.0,
        "avg_delay_frames": total_delay / hits if hits > 0 else 0,
        "false_positive_frames": false_positives,
        "per_event": results,
    }


def format_report(report: dict) -> str:
    lines = []
    lines.append("Dynamic Benchmark Report — apt_0 2400f 12 events")
    lines.append("=" * 54)
    lines.append(f"{'Event':<6} {'Type':<8} {'Frame':<7} {'Result':<8} {'Delay'}")
    lines.append("-" * 54)

    for i, r in enumerate(report["per_event"]):
        ev = r["event"]
        status = "PASS" if r["detected"] else "MISS"
        delay = f"{r['delay_frames']}f" if r["detected"] else "--"
        lines.append(
            f"{i+1:<6} {ev.get('type',''):<8} {ev.get('frame',''):<7} {status:<8} {delay}"
        )

    lines.append("-" * 54)
    lines.append(f"Recall:    {report['detected']}/{report['total_events']} ({report['recall']:.1%})")
    lines.append(f"Avg delay: {report['avg_delay_frames']:.0f} frames")
    lines.append(f"False +:   {report['false_positive_frames']} frames")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamic benchmark")
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--frame-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark_report.json"))
    args = parser.parse_args()

    events = load_event_log(args.event_log)
    metrics = load_frame_metrics(args.frame_metrics)
    report = evaluate(events, metrics)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(format_report(report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with mock data**

Create `tests/test_eval_dynamic_benchmark.py`:

```python
import json
import tempfile
from pathlib import Path
from scripts.eval_dynamic_benchmark import evaluate, load_event_log, load_frame_metrics


def test_evaluate_all_events_detected():
    """When ghost appears in detection window, event is detected."""
    events = [
        {
            "frame": 150, "type": "remove",
            "expected_judgment": "DISAPPEARED",
            "detection_window": [200, 350],
        }
    ]
    metrics = [
        {"frame_id": 100, "ghost_object_count": 0},
        {"frame_id": 200, "ghost_object_count": 0},
        {"frame_id": 210, "ghost_object_count": 1},  # ghost appears
        {"frame_id": 300, "ghost_object_count": 1},
    ]
    report = evaluate(events, metrics)
    assert report["detected"] == 1
    assert report["recall"] == 1.0
    assert report["per_event"][0]["detected"]
    assert report["per_event"][0]["detect_frame"] == 210


def test_event_missed_outside_window():
    """Ghost outside detection window is not counted as detection."""
    events = [
        {
            "frame": 150, "type": "remove",
            "expected_judgment": "DISAPPEARED",
            "detection_window": [200, 350],
        }
    ]
    metrics = [
        {"frame_id": 100, "ghost_object_count": 0},
        {"frame_id": 200, "ghost_object_count": 0},
        {"frame_id": 400, "ghost_object_count": 1},  # too late
    ]
    report = evaluate(events, metrics)
    assert report["detected"] == 0


def test_false_positive_detection():
    """Ghost outside any detection window counts as false positive."""
    events = [
        {
            "frame": 150, "type": "remove",
            "expected_judgment": "DISAPPEARED",
            "detection_window": [200, 350],
        }
    ]
    metrics = [
        {"frame_id": 100, "ghost_object_count": 1},  # before event!
        {"frame_id": 500, "ghost_object_count": 1},  # after window
    ]
    report = evaluate(events, metrics)
    assert report["false_positive_frames"] == 2


def test_move_detection():
    """MOVED event detected via moved_this_frame."""
    events = [
        {
            "frame": 750, "type": "move",
            "expected_judgment": "MOVED",
            "detection_window": [800, 950],
        }
    ]
    metrics = [
        {"frame_id": 700, "moved_this_frame": [], "ghost_object_count": 0},
        {"frame_id": 850, "moved_this_frame": [5], "ghost_object_count": 0},
    ]
    report = evaluate(events, metrics)
    assert report["detected"] == 1
```

Run:
```bash
python3 -m pytest tests/test_eval_dynamic_benchmark.py -v
```
Expected: 4 passed

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_dynamic_benchmark.py tests/test_eval_dynamic_benchmark.py
git commit -m "feat: add dynamic benchmark eval script with tests"
```

---

### Task 6: Run Stage 2 — pipeline on generated dataset

**Files:**
- Creates: `outputs/dynamic_benchmark/pipeline_output/frame_metrics.jsonl`
- Creates: `outputs/dynamic_benchmark/benchmark_report.json`

- [ ] **Step 1: Create pipeline run config for apt_0 dataset**

Create `configs/apt0_dynamic_benchmark.yaml` by copying `configs/default.yaml` and ensuring `dynamic_maintenance` has TSDF-driven params.

- [ ] **Step 2: Run pipeline on generated dataset**

```bash
python3 run_room0_full_eval.py \
  --dataset-root outputs/dynamic_benchmark/dataset \
  --num-frames 0 \
  --config-path configs/apt0_dynamic_benchmark.yaml \
  --proposal-backend sam2 \
  --proposal-device cuda \
  --sam-repo-root /home/ww/vv/paper2/DovSG/third_party/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/paper2/DovSG/checkpoints/segment-anything-2 \
  --sam-encoder hiera_l \
  --experiment-name dynamic_benchmark_apt0
```

Copy frame_metrics to benchmark output dir:
```bash
cp outputs/dynamic_benchmark_apt0/room0/frame_metrics.jsonl \
   outputs/dynamic_benchmark/pipeline_output/frame_metrics.jsonl
```

- [ ] **Step 3: Run eval script**

```bash
python3 scripts/eval_dynamic_benchmark.py \
  --event-log outputs/dynamic_benchmark/dataset/event_log.json \
  --frame-metrics outputs/dynamic_benchmark/pipeline_output/frame_metrics.jsonl \
  --output outputs/dynamic_benchmark/benchmark_report.json
```
Expected: terminal output with per-event PASS/MISS and recall summary.

- [ ] **Step 4: Commit results**

```bash
git add outputs/dynamic_benchmark/benchmark_report.json
git commit -m "feat: dynamic benchmark results — apt_0 12 events"
```

---
