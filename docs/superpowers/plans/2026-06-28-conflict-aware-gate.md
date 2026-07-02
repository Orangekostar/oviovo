# Conflict-Aware Gate 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 改 `_filter_patch_by_surface_owner`：voxel owner 冲突时不直接拒绝，而是检查旧 owner 的 TSDF support 是否在衰退 + 新 patch 是否持续出现，以此判断真实变化。

**Architecture:** 在 surface_owner_gate 的 foreign_owner 分支加冲突感知逻辑。

**Tech Stack:** Python 3, numpy

## Global Constraints

- 仅改 `src/modules/object_update.py` 的 `_filter_patch_by_surface_owner`
- 不改 gate 的默认行为（静态场景保持严格）
- support_declining = owned_voxel_count < peak_voxel_count * 0.5
- conflict_hits >= 2 才开门

---

### Task 1: Implement conflict-aware gate

**Files:**
- Modify: `src/modules/object_update.py:_filter_patch_by_surface_owner`

- [ ] **Step 1: Read current foreign_owner logic**

Find the section where `decision_foreign_owner_mask` is set (around line 679-695). Understand the current flow.

- [ ] **Step 2: Add conflict tracking dict and logic**

Before the decision loop, initialize:
```python
conflict_hits: Dict[Tuple[Tuple[int,int,int], int], int] = {}
```

In the foreign_owner detection block (where `owner_id >= 0` and `owner_id != allowed_owner_id`), replace:
```python
elif owner_id >= 0:
    decision_foreign_owner_mask[decision_idx] = True
```

With:
```python
elif owner_id >= 0:
    # Conflict-aware: check if old owner's support is declining
    if allowed_owner_id is not None:
        old_support = self.tsdf_module.summarize_instance_support(
            state.tsdf_volume, owner_id
        )
        old_owned = old_support["owned_voxel_count"]
        old_peak = max(old_owned, 1)
        support_declining = old_owned < old_peak * 0.5

        if support_declining:
            conflict_key = (key, int(allowed_owner_id))
            conflict_hits[conflict_key] = conflict_hits.get(conflict_key, 0) + 1
            if conflict_hits[conflict_key] >= 2:
                continue  # ACCEPT: old owner fading, new evidence persistent

    decision_foreign_owner_mask[decision_idx] = True
```

- [ ] **Step 3: Verify existing tests pass**

```bash
python3 -m pytest tests/ -x -k "object_update" 2>&1 | tail -5
```

- [ ] **Step 4: Re-run dynamic benchmark with stride=10**

```bash
cd /home/ww/tmp/oviovo-pr && python3 run_room0_full_eval.py \
  --dataset-root outputs/dynamic_benchmark_v4/full --num-frames 0 --frame-stride 10 \
  --config-path configs/apt0_dynamic_benchmark.yaml \
  --proposal-backend sam2 --proposal-device cuda \
  --sam-repo-root /home/ww/vv/paper2/DovSG/third_party/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/paper2/DovSG/checkpoints/segment-anything-2 \
  --sam-encoder hiera_l --experiment-name eval_conflict_gate
```

- [ ] **Step 5: Evaluate and compare**

```bash
python3 -c "
import json
for name,path in [('TSDF', 'outputs/eval_conflict_gate/room0/frame_metrics.jsonl'),
                   ('Time10', 'outputs/eval_new_s10_time/room0/frame_metrics.jsonl')]:
    ghost=0;moved=0;fp=0
    with open(path) as f:
        for line in f:
            d=json.loads(line);fid=d['frame_id']
            g=d.get('ghost_object_count',0)
            if g>0:
                if 85<=fid<=100: ghost+=1
                elif fid<80: fp+=1
            if len(d.get('moved_this_frame',[]))>0 and 85<=fid<=100: moved+=1
    print(f'{name}: DELETE={ghost} MOVE={moved} FP={fp}')
"
```

Expected: TSDF 显示非零 DELETE/MOVE，FP 接近 0，Time 保持不变。

- [ ] **Step 6: Commit**

```bash
git add src/modules/object_update.py
git commit -m "feat: conflict-aware surface owner gate"
```

---