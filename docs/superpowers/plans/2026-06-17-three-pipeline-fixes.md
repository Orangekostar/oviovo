# Three Pipeline Bottleneck Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three code-level anti-patterns in the top pipeline bottlenecks without changing any algorithm logic — all three produce identical results to baseline.

**Architecture:** (1) Reorder checks in `_score_pair` so cheap label comparisons short-circuit before expensive geometry computation. (2) Revert the failed vectorized surface_owner_gate, restore original Python loop. (3) Lower the geometry NN vectorized threshold so more pairs use kd-tree instead of brute-force O(N×M) array allocation.

**Tech Stack:** Python 3.10, numpy, scipy.spatial.cKDTree (existing)

## Global Constraints

- All three fixes are zero-precision-impact — evaluation results identical to baseline
- Each fix is independently testable with a 200f s=10 run
- No new config keys needed (Opt 2 removes the packed-index code, Opt 1 and 3 are pure code reordering/threshold changes)

---

### Fix 1: runtime_vis — Anchor label check before geometry scoring (1933s → ~50s)

**Files:**
- Modify: `src/modules/runtime_vis.py:830-950` (`_score_pair`)

**Problem:** `_score_pair` computes adjacency, depth continuity, plane compatibility, bbox plausibility, whole-prior score, and background conflict — all BEFORE checking whether the two proposals have incompatible anchor labels. In anchor_first_sam mode, 911K/930K pairs (98%) are rejected by label mismatch. All geometry computation on those 911K pairs is wasted.

**Fix:** Move the five anchor/semantic label checks (lines 890-906) to the TOP of `_score_pair`, before any geometry computation. These are cheap string/list comparisons.

- [ ] **Step 1: Move label checks to function start**

In `_score_pair` (line 830), after the function signature and before any geometry computation, add the early-reject block:

```python
    def _score_pair(self, feature_a, feature_b, evidences, depth):
        # --- Early-reject: cheap label/identity checks first ---
        cross_class_anchor_conflict = self._anchor_labels_conflict(feature_a.proposal, feature_b.proposal)
        missing_required_anchor_label = self._missing_required_anchor_label(feature_a.proposal, feature_b.proposal)
        required_anchor_labels_mismatch = self._required_anchor_labels_mismatch(feature_a.proposal, feature_b.proposal)
        required_anchor_identity_mismatch = self._required_anchor_identity_mismatch(feature_a.proposal, feature_b.proposal)
        semantic_blocked_residual_merge = self._semantic_blocked_residual_merge(feature_a.proposal, feature_b.proposal)

        accepted_reason = "accepted"
        if semantic_blocked_residual_merge:
            return RuntimeMergeDecision(
                mask_id_a=feature_a.proposal_id, mask_id_b=feature_b.proposal_id,
                accepted=False, accepted_reason="semantic_blocked_residual",
                final_score=0.0, rejected_due_to_background_conflict=False,
                boosted_by_whole_prior=False, merged_bbox_compactness=0.0,
                adjacency_score=0.0, depth_gap_score=0.0,
                boundary_depth_continuity=0.0, plane_compatibility=0.0,
                small_object_protection=0.0, containment_score=0.0,
                whole_prior_score=0.0, background_conflict_penalty=0.0,
                linked_object_id=None,
            )
        if missing_required_anchor_label:
            return RuntimeMergeDecision(
                mask_id_a=feature_a.proposal_id, mask_id_b=feature_b.proposal_id,
                accepted=False, accepted_reason="missing_anchor_label",
                final_score=0.0, rejected_due_to_background_conflict=False,
                boosted_by_whole_prior=False, merged_bbox_compactness=0.0,
                adjacency_score=0.0, depth_gap_score=0.0,
                boundary_depth_continuity=0.0, plane_compatibility=0.0,
                small_object_protection=0.0, containment_score=0.0,
                whole_prior_score=0.0, background_conflict_penalty=0.0,
                linked_object_id=None,
            )
        if required_anchor_labels_mismatch:
            return RuntimeMergeDecision(
                mask_id_a=feature_a.proposal_id, mask_id_b=feature_b.proposal_id,
                accepted=False, accepted_reason="anchor_label_mismatch",
                final_score=0.0, rejected_due_to_background_conflict=False,
                boosted_by_whole_prior=False, merged_bbox_compactness=0.0,
                adjacency_score=0.0, depth_gap_score=0.0,
                boundary_depth_continuity=0.0, plane_compatibility=0.0,
                small_object_protection=0.0, containment_score=0.0,
                whole_prior_score=0.0, background_conflict_penalty=0.0,
                linked_object_id=None,
            )
        if required_anchor_identity_mismatch:
            return RuntimeMergeDecision(
                mask_id_a=feature_a.proposal_id, mask_id_b=feature_b.proposal_id,
                accepted=False, accepted_reason="anchor_identity_mismatch",
                final_score=0.0, rejected_due_to_background_conflict=False,
                boosted_by_whole_prior=False, merged_bbox_compactness=0.0,
                adjacency_score=0.0, depth_gap_score=0.0,
                boundary_depth_continuity=0.0, plane_compatibility=0.0,
                small_object_protection=0.0, containment_score=0.0,
                whole_prior_score=0.0, background_conflict_penalty=0.0,
                linked_object_id=None,
            )

        # --- Now do the expensive geometry computation ---
        adjacency_score = self._adjacency_score(feature_a.bbox_xyxy, feature_b.bbox_xyxy)
        # ... rest of original _score_pair unchanged ...
```

- [ ] **Step 2: Remove duplicate label checks from original position**

Remove lines 890-936 (the original `cross_class_anchor_conflict` through `required_anchor_labels_mismatch` checks) since they are now at the function start.

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
  --output-root outputs/tmp_validation --experiment-name _smoke_fix1 \
  2>&1 | tail -5
```

- [ ] **Step 4: Run 200f s=10 validation**

```bash
EXP="_20260617_fix1_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | grep -E "frame=1990|runtime_vis:|mIoU|Saved"
```

Verify: `runtime_vis` total drops dramatically; mIoU identical to baseline.

---

### Fix 2: surface_owner_gate — Revert failed vectorization, keep simple loop (2529s → ~2470s)

**Files:**
- Modify: `src/modules/object_update.py:810-870` (owner lookup loop)
- Modify: `src/modules/object_update.py:255` (remove invalidation call)
- Modify: `src/modules/object_update.py:1326-1361` (remove packed index builders)

**Problem:** The vectorized surface_owner_gate code adds per-frame dict→array conversion overhead without meaningful speedup because (a) the Python loop is only ~9000 iterations/frame and (b) dict-to-sorted-array conversion costs more than the loop savings.

**Fix:** Revert to the original simple Python for-loop (pre-vectorization), which does tuple-key creation + dict lookup per voxel. Keep the `candidate_evidence` bypass logic.

- [ ] **Step 1: Revert the owner lookup loop to pre-vectorization version**

Replace lines 810-870 (the vectorized owner lookup block with packed indexes) with the original simple loop from the candidate_evidence version:

```python
        decision_same_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_foreign_owner_mask = np.zeros(len(decision_voxels), dtype=bool)
        decision_background_owner_mask = np.zeros(len(decision_voxels), dtype=bool)

        candidate_claim = state.tsdf_volume.candidate_claim if self.candidate_evidence_enabled else None
        evidence_threshold = self.candidate_evidence_threshold

        owner_lookup_start = time.perf_counter()
        for decision_idx, voxel in enumerate(decision_voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))

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

        if self.candidate_evidence_enabled and new_object and anchor_label:
            self._accumulate_candidate_evidence(
                state.tsdf_volume,
                unique_voxels,
                anchor_label,
            )
```

- [ ] **Step 2: Remove the per-frame invalidation call**

In `process` (line 255), remove:
```python
self._invalidate_packed_indexes(state.tsdf_volume)
```

- [ ] **Step 3: Remove the packed index helper methods**

Remove the three methods added between `_pack_voxel` and `_local_pcd_voxel_pool_state`:
- `_build_packed_owner_index`
- `_build_packed_bg_index`
- `_invalidate_packed_indexes`

- [ ] **Step 4: Verify with 200f s=10 validation**

```bash
EXP="_20260617_fix2_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | grep -E "frame=1990|surface_owner_gate:|mIoU|Saved"
```

Verify: `surface_owner_gate` total returns to ~247s (baseline level), mIoU identical.

---

### Fix 3: association — Prefer kd-tree over brute-force vectorized NN (1964s → ~1500s)

**Files:**
- Modify: `src/modules/association.py:866-878` (`_select_geometry_nn_backend`)

**Problem:** The `auto` backend uses vectorized brute-force NN when both patch and object have ≤160 points. This creates a `(P, O, 3)` intermediate array. For P=96, O=96 that's fine, but for larger objects (wall: 2000 points), this balloons. The kd-tree path (O(P log O)) is more scalable and has caching.

**Fix:** Lower the `vectorized` threshold and default to `kdtree` for anything above small sizes. Also fix the `geometry_vectorized_max_points` config default.

- [ ] **Step 1: Change the backend selection threshold**

Replace lines 866-878 of `_select_geometry_nn_backend`:

```python
        # Lower threshold: prefer kd-tree for anything above trivial size
        # to avoid O(P×O) intermediate array allocation
        max_vectorized_points = self.geometry_vectorized_max_points
        if max_vectorized_points <= 0:
            max_vectorized_points = 64  # lowered from 160
        if (
            backend == "auto"
            and int(patch_point_count) <= max_vectorized_points
            and int(object_point_count) <= max_vectorized_points
        ):
            return "vectorized"
        return "kdtree"
```

- [ ] **Step 2: Lower the config default**

In `configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml`, change:
```yaml
      geometry_vectorized_max_points: 160
```
to:
```yaml
      geometry_vectorized_max_points: 64
```

- [ ] **Step 3: Verify with 200f s=10 validation**

```bash
EXP="_20260617_fix3_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | grep -E "frame=1990|association:|mIoU|Saved"
```

Verify: `association` total drops, mIoU identical to baseline.

---

### Task 4: Combined validation — all three fixes

- [ ] **Step 1: Run combined 200f s=10**

```bash
EXP="_20260617_fix_all_s10_200f" && rm -rf outputs/tmp_validation/$EXP && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 --fast-eval --lightweight-benchmark \
  --sam-version 2.1 --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation --experiment-name "$EXP" \
  2>&1 | tail -15
```

- [ ] **Step 2: Compare stage timings and mIoU against baseline**

```bash
echo "=== Combined fixes vs baseline ==="
for tag in "runtime_vis" "object_update_surface_owner_gate" "association"; do
  old=$(grep "\`$tag\`" outputs/tmp_validation/_20260617_candidate_ev_s10_200f/room0/run_report.md | grep -o "total \`[0-9.]*" | grep -o "[0-9.]*")
  new=$(grep "\`$tag\`" outputs/tmp_validation/_20260617_fix_all_s10_200f/room0/run_report.md | grep -o "total \`[0-9.]*" | grep -o "[0-9.]*")
  delta=$(python3 -c "print(round($new - $old, 1))")
  echo "$tag: ${old}s -> ${new}s ($delta)"
done
diff <(cat outputs/tmp_validation/_20260617_candidate_ev_s10_200f/replica/results.json) \
     <(cat outputs/tmp_validation/_20260617_fix_all_s10_200f/replica/results.json) && \
     echo "mIoU: IDENTICAL" || echo "mIoU: DIFFERS"
```

Expected: runtime_vis drops ~70%, association drops ~20%, surface_owner_gate returns to baseline. mIoU identical.

- [ ] **Step 3: If 200f results positive, launch 2000f s=1 full validation**

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
  --experiment-name _20260617_fix_all_s1_2000f \
  > outputs/tmp_validation/_20260617_fix_all_s1_2000f.log 2>&1 &
```
