# Evaluation Mode Switch 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 dynamic_maintenance.py 加 mode 开关支持 TSDF/time 双模式，更新 YOLO 配置，产出 AAAI 评估对比表。

**Architecture:** 配置驱动的 mode 开关（`mode: "tsdf" | "time"`），不改 pipeline。同一 V4 数据集跑多组实验，eval 脚本对比 frame_metrics。

**Tech Stack:** Python 3, pytest

## Global Constraints

- Pipeline 核心逻辑不改
- TSDF 三级判决逻辑不改
- V4 数据集不变
- YOLO 权重路径: `/home/ww/vv/paper2/yolov8s-world.pt`
- 时间模式阈值: 10/30/60 帧

---

### Task 1: Add mode switch to dynamic_maintenance.py

**Files:**
- Modify: `src/modules/dynamic_maintenance.py`

- [ ] **Step 1: Add mode to __init__ and time-based process method**

In `__init__`, add mode:

```python
self.mode = config.get("mode", "tsdf")
if self.mode == "time":
    self.ghost_max_inactive = config.get("ghost_max_inactive_frames", 30)
```

Rename current `process` to `_process_tsdf`, add dispatcher:

```python
def process(self, state: SystemState) -> SystemState:
    self.last_moved_object_ids.clear()
    self.last_new_candidate_ids.clear()
    if state.frame_count % self.check_interval != 0:
        return state
    if self.mode == "time":
        return self._process_time_based(state)
    return self._process_tsdf(state)
```

Add time-based method:

```python
def _process_time_based(self, state: SystemState) -> SystemState:
    """Original time-signal based lifecycle management."""
    max_inactive = self.ghost_max_inactive
    to_remove = []
    for obj_id, obj in state.objects.items():
        if obj.state in (ObjectState.REMOVED, ObjectState.GHOST):
            to_remove.append(obj_id)
            continue
        if obj.state != ObjectState.ACTIVE:
            continue
        frames_since = state.frame_count - obj.last_seen_frame
        if frames_since > max_inactive:
            obj.state = ObjectState.GHOST
            to_remove.append(obj_id)
    for obj_id in to_remove:
        if obj_id in state.objects:
            del state.objects[obj_id]
    return state
```

- [ ] **Step 2: Run unit tests to verify both modes work**

```bash
python3 -m pytest tests/test_dynamic_maintenance.py -x -v 2>&1 | tail -5
```
Expected: existing tests pass

- [ ] **Step 3: Write test for time mode**

In `tests/test_dynamic_maintenance.py`, add:

```python
class TestTimeMode:
    def test_time_mode_ghost_after_max_inactive(self):
        import src.core.data_structures as d
        state = d.SystemState(); state.frame_count = 100
        obj = d.ObjectMap(object_id=1, state=d.ObjectState.ACTIVE, last_seen_frame=50)
        state.objects[1] = obj
        m = DynamicMaintenanceModule({"mode":"time","ghost_max_inactive_frames":30,"lifecycle_check_interval":1})
        result = m.process(state)
        assert result.objects[1].state == d.ObjectState.GHOST  # 50 frames since last seen > 30

    def test_time_mode_active_when_recently_seen(self):
        import src.core.data_structures as d
        state = d.SystemState(); state.frame_count = 100
        obj = d.ObjectMap(object_id=1, state=d.ObjectState.ACTIVE, last_seen_frame=80)
        state.objects[1] = obj
        m = DynamicMaintenanceModule({"mode":"time","ghost_max_inactive_frames":30,"lifecycle_check_interval":1})
        result = m.process(state)
        assert result.objects[1].state == d.ObjectState.ACTIVE  # 20 frames < 30
```

```bash
python3 -m pytest tests/test_dynamic_maintenance.py::TestTimeMode -v
```
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add src/modules/dynamic_maintenance.py tests/test_dynamic_maintenance.py
git commit -m "feat: add mode switch (tsdf/time) to dynamic maintenance"
```

---

### Task 2: Update config files

**Files:**
- Modify: `configs/default.yaml`
- Modify: `configs/apt0_dynamic_benchmark.yaml`

- [ ] **Step 1: Add mode to default.yaml**

```yaml
dynamic_maintenance:
  mode: "tsdf"
  lifecycle_check_interval: 10
  # ... existing tsdf params ...
  ghost_max_inactive_frames: 30  # for time mode
```

- [ ] **Step 2: Enable YOLO in apt0 config**

```yaml
anchor_frontend:
  enabled: true
  model_path: "/home/ww/vv/paper2/yolov8s-world.pt"
  # keep other params
```

- [ ] **Step 3: Commit**

```bash
git add configs/default.yaml configs/apt0_dynamic_benchmark.yaml
git commit -m "feat: add maintenance mode + YOLO path to configs"
```

---

### Task 3: Run TSDF benchmark with YOLO enabled

- [ ] **Step 1: Run 1000f pipeline with TSDF mode + YOLO**

```bash
cd /home/ww/tmp/oviovo-pr && python3 run_room0_full_eval.py \
  --dataset-root outputs/dynamic_benchmark_v4/full --num-frames 1000 \
  --config-path configs/apt0_dynamic_benchmark.yaml \
  --proposal-backend sam2 --proposal-device cuda \
  --sam-repo-root /home/ww/vv/paper2/DovSG/third_party/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/paper2/DovSG/checkpoints/segment-anything-2 \
  --sam-encoder hiera_l --experiment-name eval_tsdf_yolo
```

- [ ] **Step 2: Verify and copy output**

```bash
cp outputs/eval_tsdf_yolo/room0/frame_metrics.jsonl outputs/eval/tsdf_yolo.jsonl
```

---

### Task 4: Run time-based benchmarks (3 configs)

- [ ] **Step 1: Run time=10**

```bash
# Edit apt0 config: mode: "time", ghost_max_inactive_frames: 10
# Run pipeline → eval_time10
cp outputs/eval_time10/room0/frame_metrics.jsonl outputs/eval/time10.jsonl
```

- [ ] **Step 2: Run time=30 and time=60**

Same flow, changing ghost_max_inactive_frames each time.

---

### Task 5: Produce comparison table

- [ ] **Step 1: Run compare script**

```bash
python3 scripts/eval_dynamic_benchmark.py --mode compare \
  --files outputs/eval/tsdf_yolo.jsonl outputs/eval/time10.jsonl outputs/eval/time30.jsonl outputs/eval/time60.jsonl \
  --event-log outputs/dynamic_benchmark_v4/full/event_log.json
```

Expected output:

| Method | DELETE | MOVE | ADD | 误判 |
|--------|--------|------|-----|------|
| TSDF (ours) | X | X | X | X |
| Time (10f) | X | X | X | X |
| Time (30f) | X | X | X | X |
| Time (60f) | X | X | X | X |

- [ ] **Step 2: Commit results**

```bash
git add outputs/eval/
git commit -m "feat: AAAI evaluation — TSDF vs time baseline comparison"
```

---