# Anchor Guided SAM Fusion Acceleration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `anchor_guided_sam.build_proposals()` overhead from ~2.7s/frame to ~0.2s/frame by eliminating redundant mask AND operations, without changing any algorithm logic.

**Architecture:** Three micro-optimizations in `src/modules/anchor_guided_sam.py`:
1. bbox pre-filter: skip full-mask AND for non-overlapping anchor-proposal pairs (4 float comparisons vs 307K bool ops)
2. Replace `np.logical_and(mask, anchor_mask).sum()` with `mask[y1:y2, x1:x2].sum()` (slice is always faster than boolean AND of two full arrays)
3. Remove anchor_mask dict entirely — anchor masks are just rectangles, compute overlap area directly from bbox coordinates

All changes are internal to `AnchorGuidedSAMModule` — no interface changes, no config changes, no evaluation changes.

**Tech Stack:** Python, numpy, existing `Proposal2D` / `Anchor2D` / `MaskAnchorRelation` types.

## Global Constraints

- Must produce identical output (same proposals, same assignments) as current implementation
- No new dependencies
- Must pass existing test suite: `python -m pytest tests/ -x -q`
- 200f stride=10 smoke test must complete with mIoU within 0.001 of baseline

---

### Task 1: Pre-compute proposal areas and bboxes outside the anchor loop

**Files:**
- Modify: `src/modules/anchor_guided_sam.py:80-100`

**Interfaces:**
- Consumes: `valid_sam_proposals` (list of `Proposal2D`)
- Produces: `prop_area`, `prop_bbox` (dicts keyed by proposal_id), used in Task 2

- [ ] **Step 1: Write benchmark script to baseline current timing**

```python
# outputs/tmp_validation/_bench_ags.py
import time, numpy as np, sys
sys.path.insert(0, '/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates')
from src.modules.anchor_guided_sam import AnchorGuidedSAMModule
from src.core.data_structures import Anchor2D, Proposal2D, Frame

# Simulate realistic input: 67 proposals, 26 anchors
rng = np.random.default_rng(42)
proposals = []
for i in range(67):
    x1, y1 = rng.integers(0, 500, 2); x2, y2 = x1 + rng.integers(20, 200), y1 + rng.integers(20, 200)
    mask = np.zeros((480, 640), dtype=bool)
    mask[y1:y2, x1:x2] = rng.random((y2-y1, x2-x1)) > 0.3
    proposals.append(Proposal2D(proposal_id=i, mask=mask, bbox_xyxy=np.array([x1,y1,x2,y2], dtype=np.float32), area=int(mask.sum()), confidence=0.8, backend_name="sam2", metadata={}))

anchors = []
for i in range(26):
    x1, y1 = rng.integers(0, 500, 2); x2, y2 = x1 + rng.integers(20, 200), y1 + rng.integers(20, 200)
    anchors.append(Anchor2D(anchor_id=i, bbox_xyxy=np.array([x1,y1,x2,y2], dtype=np.float32), class_name=f"class_{i%10}", confidence=0.7))

class FakeFrame: depth = np.zeros((480,640)); source_frame_id = 0; frame_id = 0
frame = FakeFrame()

mod = AnchorGuidedSAMModule({"enabled": True, "include_unknown_residuals": True, "include_anchor_box_fallbacks": True})

# Warmup
for _ in range(5): mod.build_proposals(frame=frame, anchors=anchors, sam_proposals=proposals)

# Timed
times = []
for _ in range(30):
    t0 = time.perf_counter()
    mod.build_proposals(frame=frame, anchors=anchors, sam_proposals=proposals)
    times.append(time.perf_counter() - t0)

print(f"Current mean: {np.mean(times)*1000:.1f} ms, median: {np.median(times)*1000:.1f} ms")
```

- [ ] **Step 2: Run benchmark to record baseline**

```bash
docker exec ww-ai python3 /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/_bench_ags.py
```

Expected output: ~2000-3000 ms (simulated 1742 mask AND calls).

- [ ] **Step 3: Pre-compute proposal areas and bboxes**

In `build_proposals()`, after line 84 (`dropped_proposal_count = ...`), add:

```python
# Pre-compute proposal areas and bboxes (used in hot loop, avoid repeated computation)
prop_area = {int(p.proposal_id): int(np.asarray(p.mask, dtype=bool).sum()) for p in valid_sam_proposals}
prop_bbox = {int(p.proposal_id): np.asarray(p.bbox_xyxy, dtype=np.float32) for p in valid_sam_proposals}
```

- [ ] **Step 4: Commit**

```bash
git add src/modules/anchor_guided_sam.py
git commit -m "perf: pre-compute proposal areas and bboxes outside anchor loop"
```

---

### Task 2: Add bbox overlap pre-filter to skip unnecessary mask ANDs

**Files:**
- Modify: `src/modules/anchor_guided_sam.py:93-98` (first loop)
- Modify: `src/modules/anchor_guided_sam.py:226-262` (`_classify_relation`)

**Interfaces:**
- Consumes: `prop_area`, `prop_bbox` (from Task 1)
- Produces: bbox-filtered `_classify_relation` that skips non-overlapping pairs

- [ ] **Step 1: Add static bbox overlap helper**

```python
@staticmethod
def _bboxes_overlap(bbox_a: np.ndarray, bbox_b: np.ndarray) -> bool:
    """True if two xyxy bboxes overlap (non-zero intersection area)."""
    ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
    bx1, by1, bx2, by2 = [float(v) for v in bbox_b]
    return ax2 > bx1 and bx2 > ax1 and ay2 > by1 and by2 > ay1
```

- [ ] **Step 2: Add bbox pre-filter in the first anchor loop**

Change lines 93-98 from:

```python
for anchor in anchors:
    relations = [
        self._classify_relation(proposal, anchor, anchor_masks[int(anchor.anchor_id)])
        for proposal in valid_sam_proposals
        if int(proposal.area) >= self.proposal_min_area
    ]
```

To:

```python
for anchor in anchors:
    aid = int(anchor.anchor_id)
    anchor_bbox = np.asarray(anchor.bbox_xyxy, dtype=np.float32)
    anchor_area = float((anchor_bbox[2] - anchor_bbox[0]) * (anchor_bbox[3] - anchor_bbox[1]))
    relations = []
    for proposal in valid_sam_proposals:
        pid = int(proposal.proposal_id)
        if prop_area.get(pid, 0) < self.proposal_min_area:
            continue
        # Bbox pre-filter: skip full-mask AND if bboxes don't overlap
        if not self._bboxes_overlap(anchor_bbox, prop_bbox[pid]):
            continue
        # Fast overlap via bbox-bounded slice (anchor mask is just a rectangle)
        relation = self._classify_relation_fast(
            proposal, anchor, anchor_bbox, anchor_area, prop_area[pid], prop_bbox[pid]
        )
        relations.append(relation)
    relations_by_anchor[aid] = relations
```

- [ ] **Step 3: Implement `_classify_relation_fast` that avoids full mask AND**

```python
def _classify_relation_fast(
    self,
    proposal: Proposal2D,
    anchor: Anchor2D,
    anchor_bbox: np.ndarray,
    anchor_area: float,
    proposal_area: int,
    proposal_bbox: np.ndarray,
) -> MaskAnchorRelation:
    """Classify relation using bbox-bounded slice instead of full-mask AND."""
    proposal_mask = np.asarray(proposal.mask, dtype=bool)
    
    # Intersection bbox
    ax1, ay1, ax2, ay2 = [float(v) for v in anchor_bbox]
    px1, py1, px2, py2 = [float(v) for v in proposal_bbox]
    ix1 = max(0, int(math.floor(max(ax1, px1))))
    iy1 = max(0, int(math.floor(max(ay1, py1))))
    ix2 = min(proposal_mask.shape[1], int(math.ceil(min(ax2, px2))))
    iy2 = min(proposal_mask.shape[0], int(math.ceil(min(ay2, py2))))
    
    if ix2 <= ix1 or iy2 <= iy1:
        overlap_area = 0
    else:
        # Slice-based overlap: proposal mask inside anchor bbox
        overlap_area = int(proposal_mask[iy1:iy2, ix1:ix2].sum())
    
    proposal_area_val = max(proposal_area, 1)
    anchor_area_val = max(anchor_area, 1)
    proposal_anchor_coverage = float(overlap_area / proposal_area_val)
    anchor_proposal_coverage = float(overlap_area / anchor_area_val)
    bbox_iou = self._bbox_iou(np.asarray(proposal.bbox_xyxy), np.asarray(anchor.bbox_xyxy))
    
    if (
        self.contained_residual_enabled
        and proposal_anchor_coverage >= self.contained_min_proposal_coverage
        and anchor_proposal_coverage <= self.contained_max_anchor_coverage
    ):
        relation = "contained_residual"
    elif (
        proposal_anchor_coverage >= self.min_proposal_anchor_coverage_for_label
        and anchor_proposal_coverage >= self.min_anchor_proposal_coverage_for_label
    ):
        relation = "scale_compatible"
    else:
        relation = "unmatched"
    
    return MaskAnchorRelation(
        proposal=proposal, anchor=anchor,
        overlap_area=overlap_area,
        proposal_anchor_coverage=proposal_anchor_coverage,
        anchor_proposal_coverage=anchor_proposal_coverage,
        bbox_iou=bbox_iou,
        relation=relation,
    )
```

- [ ] **Step 4: Run benchmark to verify speedup**

```bash
docker exec ww-ai python3 /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/_bench_ags.py
```

Expected: ~150-300 ms (vs ~2000-3000 ms baseline).

- [ ] **Step 5: Commit**

```bash
git add src/modules/anchor_guided_sam.py
git commit -m "perf: add bbox pre-filter to skip redundant mask AND in anchor-guided fusion"
```

---

### Task 3: Remove anchor_masks dict (no longer needed)

**Files:**
- Modify: `src/modules/anchor_guided_sam.py:86-89` (remove `anchor_masks` creation)
- Modify: `src/modules/anchor_guided_sam.py:110,147,155` (remove `anchor_mask` references in second loop)
- Modify: `src/modules/anchor_guided_sam.py:264-296` (`_build_anchored_proposal`)

**Interfaces:**
- Consumes: `anchor_bbox`, `anchor_area` computed in Task 2 loop (pass through to second loop)
- Produces: Cleaned-up `_build_anchored_proposal` that takes `anchor_bbox` instead of `anchor_mask`

- [ ] **Step 1: Remove `anchor_masks` creation (line 86-89)**

Delete:
```python
anchor_masks = {
    int(anchor.anchor_id): self._bbox_mask(frame, anchor.bbox_xyxy)
    for anchor in anchors
}
```

Replace with storing bbox+area per anchor for later use:

```python
anchor_info: dict[int, tuple[np.ndarray, float]] = {}
```

In the first loop (Task 2), after computing `anchor_bbox` and `anchor_area`, add:
```python
anchor_info[aid] = (anchor_bbox, anchor_area)
```

- [ ] **Step 2: Update `_build_anchored_proposal` to use bbox instead of full mask**

Change signature from:
```python
def _build_anchored_proposal(self, anchor, relations, anchor_mask):
```
To:
```python
def _build_anchored_proposal(self, anchor, relations, anchor_bbox, anchor_area):
```

Change body: replace `anchor_mask` usage with bbox-bounded operations:
```python
ax1, ay1, ax2, ay2 = [int(math.floor(float(v))) for v in anchor_bbox]
mask = np.zeros_like(np.asarray(relations[0].proposal.mask, dtype=bool), dtype=bool)
for relation in relations:
    mask = np.logical_or(mask, np.asarray(relation.proposal.mask, dtype=bool))
if self.clip_to_anchor_box:
    mask[:ay1, :] = False; mask[ay2:, :] = False
    mask[:, :ax1] = False; mask[:, ax2:] = False
```

- [ ] **Step 3: Update `_build_anchor_box_fallback` similarly**

Change signature to use `anchor_bbox` instead of `anchor_mask`. The fallback mask is just the anchor rectangle:
```python
def _build_anchor_box_fallback(self, anchor, anchor_bbox):
    ax1, ay1, ax2, ay2 = [int(math.floor(float(v))) for v in anchor_bbox]
    mask = np.zeros((480, 640), dtype=bool)
    mask[ay1:ay2, ax1:ax2] = True
    ...
```

- [ ] **Step 4: Update call sites in second loop (lines 110, 147, 155)**

```python
anchor_bbox, anchor_area = anchor_info[anchor_id]
```
Pass `anchor_bbox, anchor_area` to `_build_anchored_proposal` and `anchor_bbox` to `_build_anchor_box_fallback`.

- [ ] **Step 5: Run correctness smoke test**

```bash
docker exec ww-ai python3 -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from src.modules.anchor_guided_sam import AnchorGuidedSAMModule
from src.core.data_structures import Anchor2D, Proposal2D

# Create a simple frame with one anchor and one proposal
mask1 = np.zeros((480,640), dtype=bool); mask1[100:200, 100:300] = True
p = Proposal2D(proposal_id=0, mask=mask1, bbox_xyxy=np.array([100,100,300,200],dtype=np.float32), area=20000, confidence=0.9, backend_name='sam2', metadata={})
a = Anchor2D(anchor_id=0, bbox_xyxy=np.array([80,80,320,220],dtype=np.float32), class_name='chair', confidence=0.8)
class F: depth = np.zeros((480,640)); source_frame_id=0; frame_id=0

mod = AnchorGuidedSAMModule({'enabled':True,'include_unknown_residuals':True,'include_anchor_box_fallbacks':True,'clip_to_anchor_box':True})
proposals, assignments, debug = mod.build_proposals(frame=F(), anchors=[a], sam_proposals=[p])
print(f'Output proposals: {len(proposals)}, assignments: {len(assignments)}')
assert len(proposals) > 0, 'Should produce at least one proposal'
assert len(assignments) > 0, 'Should produce at least one assignment'
print('PASS')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/modules/anchor_guided_sam.py
git commit -m "perf: remove full-frame anchor masks, use bbox slices throughout"
```

---

### Task 4: End-to-end validation — 200f stride=10 smoke test

**Files:**
- No code changes — validation only

**Interfaces:**
- Consumes: All changes from Tasks 1-3
- Produces: Confirmed identical mIoU to baseline, confirmed ~0.6s proposal_generation

- [ ] **Step 1: Run 200f stride=10 comparison experiment**

```bash
docker exec ww-ai bash -c '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && \
EXP="_20260617_ags_fusion_opt_s10_200f" && rm -rf outputs/tmp_validation/$EXP 2>/dev/null && \
python3 run_room0_full_eval.py \
  --config-path configs/tmp_recall_sweep/replica_anchor_first_yoloworld_recall_thr005_4090.yaml \
  --num-frames 200 --frame-stride 10 \
  --fast-eval --lightweight-benchmark \
  --sam-version 2.1 \
  --sam-repo-root /home/ww/vv/paper2/OVO/thirdParty/segment-anything-2 \
  --sam-ckpt-path /home/ww/vv/oviovo/data/input/sam_ckpts \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --output-root outputs/tmp_validation \
  --experiment-name "$EXP" \
  2>&1 | tail -20
'
```

- [ ] **Step 2: Verify correctness**

Compare mIoU with fullsam_ce baseline (0.4797). Must be within 0.001.

```bash
docker exec ww-ai bash -c '
echo "optimized:" && cat /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/_20260617_ags_fusion_opt_s10_200f/replica/results.json
echo "baseline:" && cat /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/_20260617_fullsam_ce_s10_200f/replica/results.json
'
```

- [ ] **Step 3: Verify speedup**

```bash
docker exec ww-ai grep "proposal_generation" /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation/_20260617_ags_fusion_opt_s10_200f/room0/run_report.md
```

Expected: `mean ~0.5-0.7s` (down from 3.16s fullsam_ce).

- [ ] **Step 4: Commit benchmark and verification results**

```bash
git add outputs/tmp_validation/_bench_ags.py
git commit -m "bench: add anchor_guided_sam fusion benchmark + verification record"
```
