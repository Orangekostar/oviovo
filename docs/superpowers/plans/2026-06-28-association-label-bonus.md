# Association Label Bonus 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 association 评分加 label_match bonus，让 YOLO 同标签的 patch 和 object 能跨空间关联，解决 object 碎片化问题。

**Architecture:** `_compute_score` 加 label_bonus 项，不改 association 的核心空间逻辑。

**Tech Stack:** Python 3

---

### Task 1: Add label bonus to association scoring

**Files:**
- Modify: `src/modules/association.py:_compute_score`
- Modify: `configs/default.yaml`

- [ ] **Step 1: Read _compute_score method**

Find it around line 417-449 in `src/modules/association.py`. Note current scoring formula and weight variables.

- [ ] **Step 2: Add label_match_weight to config**

```yaml
association:
  label_match_weight: 0.15  # bonus for same-label patch-object matching
```

- [ ] **Step 3: Add label extraction helper**

```python
@staticmethod
def _get_patch_label(patch: Patch3D) -> str:
    return str(patch.metadata.get("anchor_class_name", "")).strip().lower()

@staticmethod  
def _get_object_label(obj: ObjectMap) -> str:
    if obj.semantic_memory.label_hypotheses:
        return str(obj.semantic_memory.label_hypotheses[0][0]).strip().lower()
    anchor = obj.debug.get("anchor_semantics", {})
    if isinstance(anchor, dict):
        l = anchor.get("canonical_label", "")
        if l: return str(l).strip().lower()
    return ""
```

- [ ] **Step 4: Add label bonus to scoring formula**

In `_compute_score`, before `total = ...`:

```python
label_bonus = 0.0
if self.label_match_weight > 0:
    patch_label = self._get_patch_label(patch)
    if patch_label:
        obj_label = self._get_object_label(obj)
        if obj_label and patch_label == obj_label:
            label_bonus = self.label_match_weight
```

Then:
```python
total = (
    self.w_voxel_vote * voxel_score
    + self.w_centroid * centroid_score
    + self.w_bbox * bbox_score
    + self.w_geometry * geo_score
    + label_bonus
)
```

- [ ] **Step 5: Verify existing tests pass**

```bash
python3 -m pytest tests/ -x -v -k "test" 2>&1 | tail -5
```

- [ ] **Step 6: Run benchmark with YOLO + label bonus**

```bash
cd /home/ww/tmp/oviovo-pr && python3 run_room0_full_eval.py \
  --dataset-root outputs/dynamic_benchmark_v4/full --num-frames 0 --frame-stride 10 \
  --config-path configs/apt0_dynamic_benchmark.yaml \
  --proposal-backend sam2 --proposal-device cuda \
  --sam-repo-root /home/ww/vv/paper2/DovSG/third_party/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/paper2/DovSG/checkpoints/segment-anything-2 \
  --sam-encoder hiera_l --experiment-name eval_label_bonus
```

Expected: max_objects ~50, DELETE > 0, MOVE > 0, FP = 0.

- [ ] **Step 7: Commit**

```bash
git add src/modules/association.py configs/default.yaml
git commit -m "feat: label match bonus in association scoring"
```

---