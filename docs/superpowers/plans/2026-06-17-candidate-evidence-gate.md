# Voxel Candidate Claim Evidence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-frame hard-threshold surface_owner_gate background/foreign rejection with cross-frame voxel-level candidate evidence accumulation, so thin/attached objects (basket, picture, rug) that YOLO consistently detects can survive the gate.

**Architecture:** Add `candidate_claim` dict to `TSDFInstanceVolume` — a sparse map from voxel → {anchor_class: claim_count}. In `_filter_patch_by_surface_owner`, look up accumulated evidence before classifying a voxel as background/foreign. Write evidence for all new_object patches regardless of gate outcome. Decay and cap evidence periodically to prevent unbounded growth.

**Tech Stack:** Python 3.10, numpy, existing dataclass-based TSDF volume

## Global Constraints

- Must not change the evaluation contract (same PLY outputs, same metric definitions)
- Must be gated behind a config flag (`candidate_evidence_enabled: false` by default)
- Must not increase per-frame wall-clock by more than 5%
- Must not break existing experiments that use the default config

---

## File Structure

| File | Change |
|------|--------|
| `src/core/data_structures.py:211-230` | Add `candidate_claim` + config fields to `TSDFInstanceVolume` |
| `src/modules/object_update.py:106-130` | Read new config values in `__init__` |
| `src/modules/object_update.py:732-1051` | Modify `_filter_patch_by_surface_owner`: evidence lookup, bypass, accumulation |
| `src/modules/object_update.py:~1360` | Add `_accumulate_candidate_evidence` helper |
| `src/modules/object_update.py:~2122` | Add `_decay_candidate_evidence` helper, call from `process` |
| `src/modules/dynamic_maintenance.py:31-42` | Call `_decay_candidate_evidence` from `process` (via hook on `object_update`) |
| `configs/default.yaml` | Add `candidate_evidence_*` fields under `surface_owner_gate` |
| `configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml` | Enable candidate evidence for testing |

---

### Task 1: Data Structure — Add `candidate_claim` to `TSDFInstanceVolume`

**Files:**
- Modify: `src/core/data_structures.py:219-230`

**Interfaces:**
- Produces: `TSDFInstanceVolume.candidate_claim: Dict[Tuple[int,int,int], Dict[str, int]]` — per-voxel per-class claim counts
- Produces: `TSDFInstanceVolume.candidate_claim_max_voxels: int = 100000`

- [ ] **Step 1: Add candidate_claim field to TSDFInstanceVolume**

In `src/core/data_structures.py`, after line 228 (`voxel_owner_id`), add:

```python
    # Candidate claim evidence for surface_owner_gate bypass.
    # Maps voxel → {anchor_class_name: claim_count}.
    # Written for ALL new_object patches regardless of gate outcome.
    # Decayed periodically; entries with zero total claims are deleted.
    candidate_claim: Dict[Tuple[int, int, int], Dict[str, int]] = field(default_factory=dict)
    candidate_claim_max_voxels: int = 100000
```

- [ ] **Step 2: Verify the dataclass still works**

Run:
```bash
python3 -c "
from src.core.data_structures import TSDFInstanceVolume
v = TSDFInstanceVolume()
print('candidate_claim:', type(v.candidate_claim).__name__, len(v.candidate_claim))
print('candidate_claim_max_voxels:', v.candidate_claim_max_voxels)
"
```
Expected: `candidate_claim: dict 0` and `candidate_claim_max_voxels: 100000`

- [ ] **Step 3: Commit**

```bash
git add src/core/data_structures.py
git commit -m "feat: add candidate_claim dict to TSDFInstanceVolume for voxel-level evidence"
```

---

### Task 2: Config — Add candidate evidence parameters

**Files:**
- Modify: `configs/default.yaml` (under `object_update.surface_owner_gate`)
- Modify: `configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml`

**Interfaces:**
- Produces: config keys read by `ObjectUpdateModule.__init__` in Task 3

- [ ] **Step 1: Add new config keys to default.yaml**

In `configs/default.yaml`, inside `object_update.surface_owner_gate`, after line with `self_background_margin: 0.15`:

```yaml
      candidate_evidence_enabled: false
      candidate_evidence_threshold: 2
      candidate_evidence_decay_interval: 100
      candidate_evidence_max_voxels: 100000
```

- [ ] **Step 2: Enable in the recall_thr005 test config**

In `configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml`, same location:

```yaml
      candidate_evidence_enabled: true
      candidate_evidence_threshold: 2
      candidate_evidence_decay_interval: 100
      candidate_evidence_max_voxels: 100000
```

- [ ] **Step 3: Verify config loading**

Run:
```bash
python3 -c "
import yaml
with open('configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml') as f:
    cfg = yaml.safe_load(f)
sg = cfg['object_update']['surface_owner_gate']
print('candidate_evidence_enabled:', sg.get('candidate_evidence_enabled'))
print('candidate_evidence_threshold:', sg.get('candidate_evidence_threshold'))
"
```
Expected: `True` and `2`

- [ ] **Step 4: Commit**

```bash
git add configs/default.yaml configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml
git commit -m "feat: add candidate_evidence config params to surface_owner_gate"
```

---

### Task 3: ObjectUpdateModule — Read new config and add helpers

**Files:**
- Modify: `src/modules/object_update.py:106-130` (config reading in `__init__`)
- Modify: `src/modules/object_update.py` (add helper methods near line 1360)

**Interfaces:**
- Consumes: `surface_owner_gate` config dict from Task 2
- Produces: `self.candidate_evidence_enabled: bool`
- Produces: `self.candidate_evidence_threshold: int`
- Produces: `self.candidate_evidence_decay_interval: int`
- Produces: `self.candidate_evidence_max_voxels: int`
- Produces: `_accumulate_candidate_evidence(state, patch, anchor_label)`
- Produces: `_decay_candidate_evidence(state, current_frame)`

- [ ] **Step 1: Add config reading in __init__**

After line 125 (`self.surface_gate_self_background_margin = ...`), add:

```python
        self.candidate_evidence_enabled = bool(gate_cfg.get("candidate_evidence_enabled", False))
        self.candidate_evidence_threshold = max(1, int(gate_cfg.get("candidate_evidence_threshold", 2)))
        self.candidate_evidence_decay_interval = max(10, int(gate_cfg.get("candidate_evidence_decay_interval", 100)))
        self.candidate_evidence_max_voxels = max(1000, int(gate_cfg.get("candidate_evidence_max_voxels", 100000)))
```

- [ ] **Step 2: Add _accumulate_candidate_evidence method**

After `_local_pcd_voxel_key` (around line 1414), add:

```python
    @staticmethod
    def _accumulate_candidate_evidence(
        volume: TSDFInstanceVolume,
        voxels: np.ndarray,
        unique_voxels: np.ndarray,
        anchor_label: str,
    ) -> None:
        """Increment candidate claim count for each unique voxel + anchor class.

        Called for ALL new_object patches regardless of gate outcome.
        One claim per unique voxel per frame.
        """
        if not anchor_label or len(unique_voxels) == 0:
            return
        candidate_claim = volume.candidate_claim
        for uv in unique_voxels:
            key = (int(uv[0]), int(uv[1]), int(uv[2]))
            claims = candidate_claim.get(key)
            if claims is None:
                candidate_claim[key] = {anchor_label: 1}
            else:
                claims[anchor_label] = claims.get(anchor_label, 0) + 1
```

- [ ] **Step 3: Add _decay_candidate_evidence method**

After `_accumulate_candidate_evidence`, add:

```python
    def _decay_candidate_evidence(self, state: SystemState, current_frame: int) -> None:
        """Decay candidate claim counts and remove stale entries."""
        if not self.candidate_evidence_enabled:
            return
        if current_frame % self.candidate_evidence_decay_interval != 0:
            return

        volume = state.tsdf_volume
        candidate_claim = volume.candidate_claim
        decay_factor = 0.9
        keys_to_delete = []

        for voxel_key, claims in candidate_claim.items():
            for cls in list(claims.keys()):
                claims[cls] *= decay_factor
                if claims[cls] < 0.5:
                    del claims[cls]
            if not claims:
                keys_to_delete.append(voxel_key)

        for key in keys_to_delete:
            del candidate_claim[key]

        # Cap total voxels if exceeded
        if len(candidate_claim) > self.candidate_evidence_max_voxels:
            # Drop the oldest/smallest entries: sort by total claim count, keep top max_voxels
            items = sorted(
                candidate_claim.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True,
            )
            for voxel_key, _ in items[self.candidate_evidence_max_voxels:]:
                del candidate_claim[voxel_key]

        volume.candidate_claim_max_voxels = self.candidate_evidence_max_voxels
```

- [ ] **Step 4: Verify module initializes without errors**

Run:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.pipelines.main_pipeline import Pipeline
p = Pipeline('configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml')
ou = p.object_update
print('candidate_evidence_enabled:', ou.candidate_evidence_enabled)
print('candidate_evidence_threshold:', ou.candidate_evidence_threshold)
print('candidate_evidence_decay_interval:', ou.candidate_evidence_decay_interval)
print('candidate_evidence_max_voxels:', ou.candidate_evidence_max_voxels)
"
```
Expected: `True`, `2`, `100`, `100000`

- [ ] **Step 5: Commit**

```bash
git add src/modules/object_update.py
git commit -m "feat: add candidate evidence config reading and helper methods"
```

---

### Task 4: Modify `_filter_patch_by_surface_owner` — Evidence lookup + bypass

**Files:**
- Modify: `src/modules/object_update.py:732-1051` (`_filter_patch_by_surface_owner`)
- Modify: `src/modules/object_update.py:264-442` (callers passing `state`)

**Interfaces:**
- Consumes: `state.tsdf_volume.candidate_claim` from Task 1
- Consumes: `self.candidate_evidence_enabled`, `self.candidate_evidence_threshold` from Task 3
- Consumes: `_accumulate_candidate_evidence` from Task 3

This is the core change. The owner lookup loop (lines 809-818) gets two modifications:

1. **Before classifying**: look up candidate evidence; set `bypass` flag if evidence >= threshold
2. **After classifying**: call `_accumulate_candidate_evidence` (regardless of gate outcome)

- [ ] **Step 1: Add bypass logic in the owner lookup loop**

Replace lines 808-818:

```python
        # --- existing ---
        owner_lookup_start = time.perf_counter()
        for decision_idx, voxel in enumerate(decision_voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            owner_id = int(voxel_owner_id.get(key, -1))
            if allowed_owner_id is not None and owner_id == int(allowed_owner_id):
                decision_same_owner_mask[decision_idx] = True
            elif owner_id >= 0:
                decision_foreign_owner_mask[decision_idx] = True

            if self.surface_gate_background_enabled:
                decision_background_owner_mask[decision_idx] = self_structural_background or key in background_support
        owner_lookup_sec = float(time.perf_counter() - owner_lookup_start)
```

With:

```python
        # --- new: candidate evidence lookup ---
        candidate_claim = state.tsdf_volume.candidate_claim if self.candidate_evidence_enabled else None
        evidence_threshold = self.candidate_evidence_threshold

        owner_lookup_start = time.perf_counter()
        for decision_idx, voxel in enumerate(decision_voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))

            # Check accumulated candidate evidence for this anchor class
            bypass = False
            if candidate_claim is not None and anchor_label:
                claims = candidate_claim.get(key, {})
                evidence = int(claims.get(anchor_label, 0))
                bypass = evidence >= evidence_threshold

            owner_id = int(voxel_owner_id.get(key, -1))
            if allowed_owner_id is not None and owner_id == int(allowed_owner_id):
                decision_same_owner_mask[decision_idx] = True
            elif owner_id >= 0 and not bypass:
                decision_foreign_owner_mask[decision_idx] = True

            if self.surface_gate_background_enabled and not bypass:
                decision_background_owner_mask[decision_idx] = self_structural_background or key in background_support
        owner_lookup_sec = float(time.perf_counter() - owner_lookup_start)
```

- [ ] **Step 2: Add evidence accumulation after the classification loop**

After `owner_lookup_sec = ...` and before the `decision_background_reject_mask` line (currently line 821), add:

```python
        # Accumulate candidate evidence for new_object patches (regardless of gate outcome)
        if self.candidate_evidence_enabled and new_object and anchor_label:
            self._accumulate_candidate_evidence(
                state.tsdf_volume,
                voxels,
                unique_voxels,
                anchor_label,
            )
```

- [ ] **Step 3: Verify with smoke test**

Run 3-frame test to ensure no crashes:
```bash
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 3 --frame-stride 1 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name _smoke_candidate_evidence \
  2>&1 | tail -10
```
Expected: completes without traceback, "Saved outputs"

- [ ] **Step 4: Commit**

```bash
git add src/modules/object_update.py
git commit -m "feat: add candidate evidence bypass in surface_owner_gate owner lookup"
```

---

### Task 5: Wire decay into pipeline lifecycle

**Files:**
- Modify: `src/modules/object_update.py:438-442` (end of `process` method)

**Interfaces:**
- Consumes: `_decay_candidate_evidence` from Task 3
- Consumes: `state.frame_count`

- [ ] **Step 1: Call decay at the end of object_update.process**

After line 442 (`return state`), immediately before it, add:

```python
        # Decay candidate claim evidence periodically
        self._decay_candidate_evidence(state, current_frame)
```

(Replace `current_frame` with the variable computed earlier — line 254: `current_frame = int(max(...))`)

- [ ] **Step 2: Verify with 20-frame test**

Run 20-frame test:
```bash
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 20 --frame-stride 1 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name _smoke_candidate_ev_20f \
  2>&1 | grep -E "frame=|Saved|Error"
```
Expected: processes 20 frames without errors

- [ ] **Step 3: Commit**

```bash
git add src/modules/object_update.py
git commit -m "feat: wire candidate evidence decay into object_update lifecycle"
```

---

### Task 6: Validation — 200f stride=10 comparison experiment

**Files:**
- None (config-only)

This is the definitive test: run the full 200f stride=10 experiment with candidate evidence enabled and compare against baseline.

- [ ] **Step 1: Run 200f experiment with candidate evidence**

```bash
rm -rf outputs/tmp_validation/_20260617_candidate_ev_s10_200f
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation \
  --experiment-name _20260617_candidate_ev_s10_200f \
  2>&1 | tee outputs/tmp_validation/_20260617_candidate_ev_s10_200f.log
```

- [ ] **Step 2: Compare results**

```bash
echo "=== Without candidate evidence (baseline) ==="
cat outputs/tmp_validation/20260615_yoloworld_recall_thr005_fast_s10_200f/replica/results.json
echo "=== With candidate evidence ==="
cat outputs/tmp_validation/_20260617_candidate_ev_s10_200f/replica/results.json
echo "=== Per-class delta ==="
python3 -c "
import json
with open('outputs/tmp_validation/20260615_yoloworld_recall_thr005_fast_s10_200f/replica/classes_iou.json') as f:
    old = json.load(f)
with open('outputs/tmp_validation/_20260617_candidate_ev_s10_200f/replica/classes_iou.json') as f:
    new = json.load(f)
for cls in sorted(set(list(old)+list(new))):
    d = new.get(cls,0) - old.get(cls,0)
    if abs(d) > 0.01:
        print(f'  {cls:20s}: {old.get(cls,0):.4f} -> {new.get(cls,0):.4f}  ({d:+.4f})')
"
```

Expected: basket and picture IoU should increase from 0.0. surface_owner_gate reject rate should decrease.

- [ ] **Step 3: Commit results**

```bash
git add outputs/tmp_validation/_20260617_candidate_ev_s10_200f/
git commit -m "exp: candidate evidence validation run (200f s=10)"
```

---

### Task 7: (Optional) Run full 2000f stride=1 with candidate evidence

Only if Task 6 shows positive results.

- [ ] **Step 1: Launch 2000f experiment**

```bash
nohup python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 2000 --frame-stride 1 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation \
  --experiment-name _20260617_candidate_ev_s1_2000f \
  > outputs/tmp_validation/_20260617_candidate_ev_s1_2000f.log 2>&1 &
```

- [ ] **Step 2: Compare against baseline 2000f**

After completion, compare mIoU and per-class IoU against the anchor_first_sam 2000f baseline and the observation_first 2000f baseline.

---
