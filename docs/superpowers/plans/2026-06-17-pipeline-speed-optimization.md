# Pipeline Speed Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate the 2000f pipeline by ~37% (3.5h → ~2.2h) through three zero-precision-impact optimizations: vectorize surface_owner_gate owner lookup, short-circuit runtime_vis pairwise computation in anchor_first_sam mode, and cache association geometry kd-trees.

**Architecture:** Three independent optimizations targeting the top three stage bottlenecks (surface_owner_gate 21%, runtime_vis 16%, association 16%). All three are pure algebraic equivalences or cache optimizations with zero impact on mIoU. Each optimization is self-contained in a single module.

**Tech Stack:** Python 3.10, numpy, scipy.spatial.cKDTree (existing)

## Global Constraints

- Must produce identical evaluation results (mIoU, mAcc) compared to pre-optimization baseline
- Must not change any config defaults — new config keys gated by `enabled: false` where applicable
- All three optimizations are independent — each can be shipped and validated separately
- Per-frame wall-clock overhead of any optimization must be ≤ 0.1% when disabled

---

## File Structure

| File | Change | Optimization |
|------|--------|-------------|
| `src/modules/object_update.py` | Vectorize owner lookup loop | Opt 1 |
| `src/modules/runtime_vis.py` | Add anchor_first_sam shortcut | Opt 2 |
| `src/modules/association.py` | Stabilize kd-tree cache keys | Opt 3 |

---

### Optimization 1: Vectorize surface_owner_gate owner lookup (247s → ~70s)

**Files:**
- Modify: `src/modules/object_update.py:809-850` (owner lookup loop)

**Approach:** Replace Python for-loop over decision voxels with `pack_voxel` + `np.searchsorted` batch classification. Pre-build packed int64 sorted arrays for `voxel_owner_id` and `background_support` once per frame, then use vectorized searchsorted for all decision voxels at once.

`_pack_voxel` already exists at line 1286.

- [ ] **Step 1: Add lazy packed-key index builder**

After `_pack_voxel` (line 1292), add helper methods that convert `voxel_owner_id` dict and `background_support` set to packed int64 sorted arrays with a validity flag:

```python
    @staticmethod
    def _build_packed_owner_index(volume: "TSDFInstanceVolume") -> None:
        """Build packed int64 lookup arrays for voxel_owner_id, cached on volume."""
        if getattr(volume, "_packed_owner_valid", False):
            return
        owner_dict = volume.voxel_owner_id
        if not owner_dict:
            volume._packed_owner_keys = np.array([], dtype=np.int64)
            volume._packed_owner_vals = np.array([], dtype=np.int32)
        else:
            keys_tuple = np.array(list(owner_dict.keys()), dtype=np.int64)
            volume._packed_owner_keys = ObjectUpdateModule._pack_voxel(keys_tuple)
            volume._packed_owner_vals = np.array(list(owner_dict.values()), dtype=np.int32)
            order = np.argsort(volume._packed_owner_keys)
            volume._packed_owner_keys = volume._packed_owner_keys[order]
            volume._packed_owner_vals = volume._packed_owner_vals[order]
        volume._packed_owner_valid = True

    @staticmethod
    def _build_packed_bg_index(volume: "TSDFInstanceVolume", bg_support: set) -> None:
        """Build packed int64 sorted array for background_support, cached on volume."""
        bg_hash = hash(frozenset({(k[0], k[1], k[2]) for k in list(bg_support)[:100]}))
        if getattr(volume, "_packed_bg_valid", False) and getattr(volume, "_packed_bg_hash", -1) == bg_hash:
            return
        if not bg_support:
            volume._packed_bg_keys = np.array([], dtype=np.int64)
        else:
            keys_tuple = np.array(list(bg_support), dtype=np.int64)
            volume._packed_bg_keys = np.sort(ObjectUpdateModule._pack_voxel(keys_tuple))
        volume._packed_bg_valid = True
        volume._packed_bg_hash = bg_hash

    @staticmethod
    def _invalidate_packed_indexes(volume: "TSDFInstanceVolume") -> None:
        """Call after any modification to voxel_owner_id."""
        volume._packed_owner_valid = False
        volume._packed_bg_valid = False
```

- [ ] **Step 2: Replace the owner lookup loop with vectorized classification**

Replace the loop block (lines 809-838 in the modified file) with the vectorized version:

```python
        decision_same_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_foreign_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_background_owner_mask = np.zeros(len(decision_voxels), dtype=bool)

        # Build packed indexes (lazy, cached on volume until invalidated)
        self._build_packed_owner_index(state.tsdf_volume)
        if self.surface_gate_background_enabled:
            self._build_packed_bg_index(state.tsdf_volume, background_support)

        owner_keys = state.tsdf_volume._packed_owner_keys
        owner_vals = state.tsdf_volume._packed_owner_vals
        bg_keys = state.tsdf_volume._packed_bg_keys if self.surface_gate_background_enabled else None

        # Vectorized: pack all decision voxels, batch-lookup via searchsorted
        owner_lookup_start = time.perf_counter()
        dv_packed = self._pack_voxel(decision_voxels)
        n_dv = len(dv_packed)

        # --- voxel_owner_id lookup ---
        if len(owner_keys) > 0:
            idx = np.searchsorted(owner_keys, dv_packed)
            found = (idx < len(owner_keys)) & (owner_keys[idx] == dv_packed)
            owner_id = np.where(found, owner_vals[idx], -1)
        else:
            owner_id = np.full(n_dv, -1, dtype=np.int32)

        # --- classify ---
        if allowed_owner_id is not None:
            decision_same_owner_mask = owner_id == int(allowed_owner_id)
        decision_foreign_owner_mask = (owner_id >= 0) & ~decision_same_owner_mask

        # --- candidate evidence bypass ---
        if self.candidate_evidence_enabled and anchor_label:
            candidate_claim = state.tsdf_volume.candidate_claim
            for i in range(n_dv):
                key = (int(decision_voxels[i, 0]), int(decision_voxels[i, 1]), int(decision_voxels[i, 2]))
                claims = candidate_claim.get(key, {})
                evidence = int(claims.get(anchor_label, 0))
                if evidence >= self.candidate_evidence_threshold:
                    decision_foreign_owner_mask[i] = False
                    decision_background_owner_mask[i] = False  # pre-cleared below

        # --- background lookup ---
        if self.surface_gate_background_enabled:
            if self_structural_background:
                decision_background_owner_mask[:] = True
            elif len(bg_keys) > 0:
                idx_bg = np.searchsorted(bg_keys, dv_packed)
                decision_background_owner_mask = (idx_bg < len(bg_keys)) & (bg_keys[idx_bg] == dv_packed)
            else:
                decision_background_owner_mask[:] = False
            # Re-apply bypass for background
            if self.candidate_evidence_enabled and anchor_label:
                candidate_claim = state.tsdf_volume.candidate_claim
                for i in range(n_dv):
                    key = (int(decision_voxels[i, 0]), int(decision_voxels[i, 1]), int(decision_voxels[i, 2]))
                    claims = candidate_claim.get(key, {})
                    evidence = int(claims.get(anchor_label, 0))
                    if evidence >= self.candidate_evidence_threshold:
                        decision_background_owner_mask[i] = False
        owner_lookup_sec = float(time.perf_counter() - owner_lookup_start)

        # Accumulate candidate evidence (unchanged from current)
        if self.candidate_evidence_enabled and new_object and anchor_label:
            self._accumulate_candidate_evidence(
                state.tsdf_volume,
                unique_voxels,
                anchor_label,
            )
```

- [ ] **Step 3: Add invalidation call after TSDF integration**

In `object_update.py`, after `self.tsdf_module.integrate_patch(...)` calls (lines 301, 424, 1737), add:

```python
self._invalidate_packed_indexes(state.tsdf_volume)
```

- [ ] **Step 4: Verify with 20-frame smoke test**

```bash
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 20 --frame-stride 1 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name _smoke_opt1 \
  2>&1 | tail -10
```

Expected: no traceback, "Saved outputs"

- [ ] **Step 5: Run 200f s=10 comparison**

```bash
EXP="_20260617_opt1_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | grep -E "frame=|surface_owner_gate|mIoU|Saved"
```

Verify: `surface_owner_gate` total drops from ~247s toward ~70s. All class IoUs identical to baseline within 0.01.

---

### Optimization 2: Short-circuit runtime_vis in anchor_first_sam mode (191s → ~50s)

**Files:**
- Modify: `src/modules/runtime_vis.py:210-235` (beginning of `process`)

**Approach:** In anchor_first_sam mode, all proposals have unique anchor labels and `require_same_anchor_label_for_merge: true` guarantees zero merges. Skip `_compute_mask_features` (per-proposal depth feature extraction), `_build_whole_object_evidence`, `_attach_prior_fits`, `_apply_depth_aware_gating`, and `_compute_pairwise_decisions`. Return identity output directly with placeholder profiles.

- [ ] **Step 1: Add anchor_all_unique detection**

After `process` receives raw_proposals (line 225), add the shortcut:

```python
        # Shortcut: in anchor_first_sam mode with per-label merge blocking,
        # all proposals are guaranteed isolated → skip pairwise computation.
        if self._all_anchors_unique(raw_proposals):
            output = self._identity_output(raw_proposals)
            self.last_output = output
            return output
```

- [ ] **Step 2: Implement `_all_anchors_unique` helper**

Add to `RuntimeVisModule`:

```python
    @staticmethod
    def _all_anchors_unique(raw_proposals: List[Proposal2D]) -> bool:
        """Return True if all proposals have distinct anchor labels → no merges possible."""
        if len(raw_proposals) <= 1:
            return True
        labels = []
        for p in raw_proposals:
            anchor = str((p.metadata or {}).get("anchor_class_name", "")).strip()
            if not anchor:
                return False  # unlabeled proposals might merge
            labels.append(anchor)
        return len(set(labels)) == len(labels)
```

- [ ] **Step 3: Verify with 20-frame smoke test**

```bash
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 20 --frame-stride 1 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name _smoke_opt2 \
  2>&1 | tail -10
```

- [ ] **Step 4: Run 200f s=10 comparison**

```bash
EXP="_20260617_opt2_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | grep -E "frame=|runtime_vis|mIoU|Saved"
```

Verify: `runtime_vis` total drops from ~191s toward ~50s, all IoUs identical.

---

### Optimization 3: Stabilize association kd-tree cache keys (187s → ~150s)

**Files:**
- Modify: `src/modules/association.py:383-430` (`_build_centroid_candidate_index`)
- Modify: `src/modules/association.py:700-720` (`_geometry_consistency`)

**Approach:** The existing kd-tree cache (`_association_kdtree_cache`) uses geometry hashes as keys. These hashes currently include per-frame timestamps, causing cache misses even when geometry hasn't changed. Fix: use a content-addressable hash of the association_pcd array.

- [ ] **Step 1: Fix cache key to be geometry-content-based**

In `_build_centroid_candidate_index`, around line 383, replace the cache key computation:

Current (approximate):
```python
cache_key = (object_id, tuple(obj.centroid), obj.update_count)
```

Replace with:
```python
# Use array content hash for stable caching across frames
pcd = np.asarray(obj.association_pcd, dtype=np.float32)
cache_key = (int(object_id), hash(pcd.tobytes()) if len(pcd) > 0 else -1)
```

- [ ] **Step 2: Add dirty-flag to association_pcd cache**

In `_geometry_consistency`, before building kd-tree:

```python
pcd_bytes = pcd.tobytes()
cache_key = (int(object_id), hash(pcd_bytes) if len(pcd) > 0 else -1)
cached = self._association_kdtree_cache.get(cache_key)
if cached is not None:
    tree, _pcd_cache = cached
else:
    tree = cKDTree(pcd) if len(pcd) >= self.kdtree_cache_min_object_points else None
    self._association_kdtree_cache[cache_key] = (tree, pcd)
```

- [ ] **Step 3: Verify with 200f s=10 comparison**

```bash
EXP="_20260617_opt3_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | grep -E "frame=|association[^_]|kdtree_cache|mIoU|Saved"
```

Verify: kd-tree cache hit rate increases, `association` total drops from ~187s toward ~150s. All IoUs identical.

---

### Task 4: Combined validation — 200f s=10 with all three optimizations

Run all three optimizations together and compare against baseline.

- [ ] **Step 1: Run combined 200f experiment**

Use the same config as baseline but with all three optimizations active (they are all pure code changes, no config switches needed — Opt 1 replaces the loop, Opt 2 detects anchor_first_sam automatically, Opt 3 fixes cache keys).

```bash
EXP="_20260617_opt_all_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | tee outputs/tmp_validation/${EXP}.log
```

- [ ] **Step 2: Compare stage timings and mIoU**

```bash
echo "=== Baseline (vec+evidence only) ==="
grep -E "surface_owner_gate:|runtime_vis:|association:" outputs/tmp_validation/_20260617_candidate_ev_s10_200f/room0/run_report.md | grep "mean"
echo "=== Combined optimizations ==="
grep -E "surface_owner_gate:|runtime_vis:|association:" outputs/tmp_validation/_20260617_opt_all_s10_200f/room0/run_report.md | grep "mean"
echo ""
echo "mIoU diff:"
python3 -c "
import json
with open('outputs/tmp_validation/_20260617_candidate_ev_s10_200f/replica/results.json') as f:
    old = json.load(f)
with open('outputs/tmp_validation/_20260617_opt_all_s10_200f/replica/results.json') as f:
    new = json.load(f)
for k in old: print(f'  {k}: {old[k]:.4f} -> {new[k]:.4f}')
"
```

Expected: all mIoU/mAcc within 0.001 of baseline. Stage timings significantly reduced.

- [ ] **Step 3: Launch 2000f full validation**

If combined 200f results are positive:

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
  --experiment-name _20260617_opt_all_s1_2000f \
  > outputs/tmp_validation/_20260617_opt_all_s1_2000f.log 2>&1 &
```

---
