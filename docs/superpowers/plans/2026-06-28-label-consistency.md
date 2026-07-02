# Label Consistency 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 加 voxel label voting + same-label object merge，解决 cross-frame 碎片化问题。

**Architecture:** VoxelOwnerSupport 加 label_votes，integrate_patch 同步投票，maintenance 加 label-merge 步骤。

**Tech Stack:** Python 3, numpy

---

### Task 1: Add label_votes to VoxelOwnerSupport

**Files:** `src/core/data_structures.py:188-211`

- [ ] Step 1: Add `label_votes: Dict[str, float] = field(default_factory=dict)` to VoxelOwnerSupport
- [ ] Step 2: Add `dominant_label` property (argmax of label_votes, "" if empty)
- [ ] Step 3: Commit

### Task 2: Write label votes during TSDF integration

**Files:** `src/modules/tsdf_instance_map.py:80-82`

- [ ] Step 1: In `integrate_patch`, extract `patch_label = patch.metadata.get("anchor_class_name", "")`
- [ ] Step 2: If non-empty, `support.label_votes[patch_label] = support.label_votes.get(patch_label, 0.0) + volume.support_increment`
- [ ] Step 3: Commit

### Task 3: Add label-merge step to maintenance

**Files:** `src/modules/dynamic_maintenance.py`

- [ ] Step 1: Add `_object_voxel_keys(obj, volume)` helper — returns set of voxel keys from obj.local_pcd
- [ ] Step 2: Add `_compute_dominant_labels(state)` — returns `Dict[int, str]` mapping object_id to dominant_label
- [ ] Step 3: Add `_merge_same_label_objects(state)` — groups by label, merges centroid-close pairs
- [ ] Step 4: Add `_should_merge(a, b)` — returns True if centroids < 2.0m and at least one was seen this frame
- [ ] Step 5: Add `_merge_into(state, victim_id, keeper_id)` — transfers observations + local_pcd, marks victim as GHOST
- [ ] Step 6: Call `_merge_same_label_objects(state)` at end of `_process_tsdf`
- [ ] Step 7: Verify 24 tests pass
- [ ] Step 8: Commit

### Task 4: Run benchmark and evaluate

- [ ] Step 1: Clear cache, run stride=10 with label_bonus=0.40
- [ ] Step 2: Verify max_objects drops significantly, DELETE increases

---

### 自审

- spec 3 个改动点全部覆盖
- 无占位符
- object_id 类型一致 (int)
