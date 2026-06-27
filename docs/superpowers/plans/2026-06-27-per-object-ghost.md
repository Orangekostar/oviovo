# Per-Object Ghost 标注 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 frame_metrics 的 ghost/moved 从计数值改为 per-object (id, label) 列表，评估时按标签过滤，消除瞬态 proposal 误判。

**Architecture:** dynamic_maintenance 产出 `(object_id, label)` 列表，main_pipeline 透传到 frame_metrics，eval 脚本按 OVO 标签匹配。

**Tech Stack:** Python 3, pytest

---

### Task 1: Implement per-object ghost labels

**Files:**
- Modify: `src/modules/dynamic_maintenance.py`
- Modify: `src/pipelines/main_pipeline.py`
- Modify: `scripts/eval_dynamic_benchmark.py`

- [ ] **Step 1: Add label extraction helper to DynamicMaintenanceModule**

```python
def _get_object_label(self, obj: ObjectMap) -> str:
    """Extract canonical label from an object's semantic memory."""
    if obj.semantic_memory.label_hypotheses:
        return str(obj.semantic_memory.label_hypotheses[0][0]).strip().lower()
    anchor = obj.debug.get("anchor_semantics", {})
    if isinstance(anchor, dict):
        label = anchor.get("canonical_label", "")
        if label:
            return str(label).strip().lower()
    return ""
```

- [ ] **Step 2: Change last_moved/ghost lists to store (id, label) tuples**

In `__init__`:
```python
self.last_moved_object_ids: List[Tuple[int, str]] = []
self.last_ghost_object_ids: List[Tuple[int, str]] = []
```

Clear both in `process()`.

- [ ] **Step 3: Append labels at DISAPPEARED and MOVED points**

At MOVED:
```python
self.last_moved_object_ids.append((int(obj.object_id), self._get_object_label(obj)))
```

At DISAPPEARED:
```python
self.last_ghost_object_ids.append((int(obj.object_id), self._get_object_label(obj)))
```

- [ ] **Step 4: Add ghost_objects to frame_metrics in main_pipeline.py**

Replace `"ghost_object_count": int(...)` with:
```python
"ghost_objects": [
    list(item) for item in getattr(self.dynamic_maintenance, "last_ghost_object_ids", [])
],
"moved_objects": [
    list(item) for item in getattr(self.dynamic_maintenance, "last_moved_object_ids", [])
],
"new_candidate_objects": [
    list(item) for item in getattr(self.dynamic_maintenance, "last_new_candidate_ids", [])
],
```

Keep old keys for backward compat:
```python
"ghost_object_count": len(...),
"moved_this_frame": [oid for oid, _ in ...],
```

- [ ] **Step 5: Update eval script for label-aware detection**

```python
DELETE_LABELS = {"lamp", "indoor-plant", "cushion"}
MOVE_LABELS = {"bowl", "box"}
ADD_LABELS = {"cup", "bleach", "tennis-ball", "tennis"}

def _has_label(items, target_labels):
    for oid, label in items:
        if any(t in label.lower() for t in target_labels):
            return True
    return False
```

Window detection:
```python
if 850 <= fid <= 1000:
    ghost_objs = d.get("ghost_objects", [])
    if _has_label(ghost_objs, DELETE_LABELS): ghost += 1
    moved_objs = d.get("moved_objects", [])
    if _has_label(moved_objs, MOVE_LABELS): moved += 1
```

- [ ] **Step 6: Run existing tests**

```bash
python3 -m pytest tests/test_dynamic_maintenance.py tests/test_eval_dynamic_benchmark.py -x -v 2>&1 | tail -5
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/modules/dynamic_maintenance.py src/pipelines/main_pipeline.py scripts/eval_dynamic_benchmark.py
git commit -m "feat: per-object ghost labels with label-aware eval"
```

---

### Task 2: Re-run pipelines and produce comparison table

- [ ] **Step 1: Run TSDF mode (1000f)**

```bash
cd /home/ww/tmp/oviovo-pr && python3 run_room0_full_eval.py \
  --dataset-root outputs/dynamic_benchmark_v4/full --num-frames 1000 \
  --config-path configs/apt0_dynamic_benchmark.yaml \
  --proposal-backend sam2 --proposal-device cuda \
  --sam-repo-root /home/ww/vv/paper2/DovSG/third_party/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/paper2/DovSG/checkpoints/segment-anything-2 \
  --sam-encoder hiera_l --experiment-name eval_tsdf_final
```

- [ ] **Step 2: Run time mode (10/30/60f)**

Same command with `--config-path configs/apt0_time{N}.yaml`. Copy all output to `outputs/eval/`.

- [ ] **Step 3: Run eval and produce table**

```bash
python3 scripts/eval_dynamic_benchmark.py --mode v4 \
  --event-log outputs/dynamic_benchmark_v4/full/event_log.json \
  --frame-metrics outputs/eval_tsdf_final/room0/frame_metrics.jsonl \
  --output outputs/eval/tsdf_report.json
```

Expected result:

| Method | DELETE | MOVE | FP(pre-800) |
|--------|--------|------|-------------|
| TSDF | ~4 | ~4 | ~0 |
| Time (10f) | ? | ? | ? |
| Time (30f) | ? | ? | ? |
| Time (60f) | ? | ? | ? |

- [ ] **Step 4: Commit results**

```bash
git add outputs/eval/ && git commit -m "feat: AAAI evaluation — TSDF vs time baseline"
```

---