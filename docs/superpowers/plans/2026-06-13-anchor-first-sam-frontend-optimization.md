# Anchor-First SAM Frontend Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for implementation and review. Keep changes scoped to the online frontend, proposal backend, configs, tests, and validation scripts.

**Goal:** Speed up OViOVO online mapping by replacing the current full-frame SAM proposal generation plus `anchor_guided_sam_fusion` path with a YOLOWorld anchor-first, SAM box-prompted proposal path.

**Primary target:** remove `anchor_guided_sam_fusion` from the per-frame critical path while preserving most baseline accuracy.

**Non-goals:**
- Do not reintroduce proposal cache as a required path.
- Do not use SED++ direct semantic components as the primary frontend; the latest validation showed it is faster in the frontend but loses too much proposal quality.
- Do not rewrite the TSDF/backend mapping stack unless validation proves a frontend-only replacement cannot preserve accuracy.

## Evidence From Latest Runs

Baseline YOLOWorld+SAM, room0 200f stride=10:

- `mIoU`: `0.5190`
- `f-mIoU`: `0.6052`
- wall time: `40:37.90`
- `proposal_generation`: `10.4371s/frame`
- `sam2_proposals`: `1.2140s/frame`
- `anchor_guided_sam_fusion`: `9.0352s/frame`
- `yoloworld_primary`: `0.1306s/frame`

SED++ replacement, room0 200f stride=10:

- `mIoU`: `0.3668`
- `f-mIoU`: `0.5618`
- wall time: `42:22.95`
- `proposal_generation`: `0.8448s/frame`
- `sedpp_inference`: `0.1132s/frame`
- `sedpp_postprocess`: `0.7314s/frame`
- runtime-vis debug showed many raw components, high merge pressure, and large mismatch edges; overlay quality was visibly poor for small/instance objects.

Conclusion:

- The biggest measured frontend bottleneck is not YOLOWorld.
- Full-frame SAM mask extraction is not the dominant cost by itself.
- The dominant cost is the global matching/fusion between YOLO anchors and full-frame SAM proposals.
- SED++ proves that a cheaper frontend is not enough; the proposal source must still give clean, instance-aligned masks.

## Core Hypothesis

The current path does:

1. YOLOWorld produces anchors.
2. SAM produces many full-frame masks.
3. `anchor_guided_sam_fusion` matches and merges anchors against the full SAM proposal set.

That turns proposal construction into a high-cost global association problem.

The proposed path does:

1. YOLOWorld produces anchors.
2. SAM2 receives anchor boxes as prompts.
3. Each anchor directly returns one or a small number of masks.
4. The frontend emits anchor-labeled SAM proposals directly.

This should change the expensive step from full-frame proposal matching to bounded per-anchor prompted segmentation.

## Intended Architecture

Add an anchor-first mode alongside the existing frontend:

```text
RGB-D frame
  -> YOLOWorld anchors
  -> SAM2 image embedding once
  -> batched/looped box prompts per anchor
  -> proposal bundle with inherited anchor labels
  -> existing backend association and TSDF integration
```

Expected runtime shape:

- `yoloworld_primary`: unchanged.
- `sam2_anchor_prompts`: new timing key.
- `sam2_proposals`: should mean anchor-prompted SAM, not full automatic masks.
- `anchor_guided_sam_fusion`: absent or near zero.
- `proposal_generation`: target below `2.0s/frame` on 4090 for 200f stride=10.

## Implementation Tasks

### 1. Add SAM2 Box-Prompt Backend Support

Files:

- `src/models/sam2_proposal_backend.py`
- `src/models/proposal_backend.py`
- `src/modules/proposal.py`

Required behavior:

- Import and initialize SAM2 image predictor support, for example `SAM2ImagePredictor` when available.
- Add a real `generate_proposals_for_anchors(rgb, depth, anchors, frame=None)` implementation in `SAM2ProposalBackend`.
- Set the SAM2 image embedding once per frame.
- Run box prompts for YOLOWorld anchors.
- Prefer batched box prediction if the installed SAM2 API supports it; otherwise loop anchors while reusing the image embedding.
- Return regular proposal objects compatible with the existing OViOVO backend.
- Preserve anchor metadata:
  - `anchor_id`
  - `anchor_bbox`
  - `anchor_class`
  - `anchor_confidence`
  - `proposal_source = "anchor_prompted_sam"`
  - SAM score / predicted IoU / stability score when available
- Add configurable mask selection:
  - `multimask_output`
  - maximum masks per anchor
  - minimum mask area
  - minimum anchor-box coverage
  - optional depth-valid coverage

First-pass mask choice:

- Start with `multimask_output=False` for speed.
- If quality is weak, test `multimask_output=True` and pick the best mask using SAM score plus anchor-box coverage.

### 2. Add Anchor-First Frontend Mode

Files:

- `src/pipelines/yoloworld_sam_frontend.py`
- current pipeline config/loading code that selects the frontend path

Required behavior:

- Add a mode flag such as `pipeline.yoloworld_sam.mode: anchor_first_sam` or equivalent local style.
- In this mode:
  - run YOLOWorld anchors first;
  - call `proposal.process_for_anchors(...)`;
  - skip `anchor_guided_sam.build_proposals(...)`;
  - emit a `FrameProposalBundle` using anchor-prompted SAM proposals directly.
- Record timings separately:
  - `yoloworld_primary`
  - `sam2_anchor_prompts`
  - `proposal_generation`
  - `sam2_full_frame_ran = false`
  - `anchor_guided_sam_fusion = 0.0` or omit the key consistently.
- Preserve the existing baseline mode unchanged.

### 3. Add Experiment Config

Create a config derived from the current online YOLOWorld+SAM baseline:

- `configs/replica_yoloworld_anchor_first_sam_4090.yaml`

Suggested settings:

- enable online mode, no proposal cache
- YOLOWorld anchors enabled
- SAM2 anchor-prompt mode enabled
- full-frame SAM disabled for first-pass validation
- anchor-guided fusion disabled for first-pass validation
- runtime visual debug available but off by default

Keep baseline config untouched so A/B comparison is clean.

### 4. Tests

Add focused tests before long runs.

Backend tests:

- Fake SAM2 predictor returns one mask per anchor.
- `generate_proposals_for_anchors` preserves anchor labels and scores.
- Empty anchors returns empty proposals without running full-frame SAM.
- Invalid/tiny boxes are skipped or clipped deterministically.

Pipeline tests:

- Anchor-first mode does not call `anchor_guided_sam.build_proposals`.
- Timing metadata contains `sam2_anchor_prompts`.
- `sam2_full_frame_ran` is false.
- Proposal source is `anchor_prompted_sam`.
- Baseline mode still calls the old fusion path.

Config/smoke tests:

- Config loads.
- 1-frame room0 run completes.
- Runtime-vis overlay can be generated for the new mode.

### 5. Validation Protocol

Run in this order:

1. Unit tests:

```bash
pytest tests/test_pipeline.py tests/test_proposal_backend.py -q
```

2. One-frame smoke run with runtime-vis enabled.

3. 20-frame room0 stride=10 run:

- verify no crashes;
- inspect overlays;
- compare proposal counts and class distribution;
- check `anchor_guided_sam_fusion` is gone.

4. 200-frame room0 stride=10 run:

- compare against the latest YOLOWorld+SAM baseline;
- save metrics JSON;
- save per-stage runtime JSON;
- save runtime-vis overlay samples.

Minimum 200f report fields:

- `mIoU`
- `mAcc`
- `f-mIoU`
- `f-mAcc`
- wall time
- average per-frame timings
- proposal count mean/median
- anchor count mean/median
- mask count per anchor
- skipped-anchor count
- runtime-vis overlay path

## Acceptance Gates

Hard gates for first successful replacement:

- `anchor_guided_sam_fusion <= 0.1s/frame` or absent.
- `proposal_generation <= 2.0s/frame`.
- 200f room0 stride=10 `mIoU >= 0.50`.
- 200f room0 stride=10 `f-mIoU >= 0.60`.
- Runtime-vis overlays do not show the severe small-object collapse seen in SED++.

Stretch target:

- wall time at least 20 percent faster than baseline 200f stride=10.

Reject the replacement if:

- mIoU drops below `0.47`, even if speed improves.
- overlays show consistent missing chairs/tables/small objects.
- anchor labels are visibly attached to wrong masks at high frequency.

## Fallback Branches

If SAM2 box prompts are fast but inaccurate:

- enable `multimask_output=True`;
- choose masks by SAM score plus anchor coverage;
- add depth-valid coverage filtering;
- allow up to 2 masks per large anchor.

If YOLOWorld misses too many objects:

- add sparse residual SAM only every `N` frames;
- restrict residual SAM to unexplained depth regions;
- do not restore full-frame SAM plus full fusion on every frame.

If many structural classes degrade:

- route wall/floor/ceiling/window/door/blinds through a structural path;
- keep object instances on anchor-prompted SAM;
- compare structural routing separately because it can improve f-mIoU while hiding object failures.

If SAM2 predictor API is unavailable in the current container:

- update the compatibility environment first;
- install the SAM2 package version that exposes image predictor box prompts;
- only use Hugging Face/mirror downloads for weights, not code paths that change runtime semantics.

## Review Checklist

The review agent should verify:

- No proposal cache dependency was introduced.
- Baseline YOLOWorld+SAM behavior is unchanged.
- New mode records enough timing data to prove where time moved.
- New proposal metadata is sufficient for runtime-vis and backend debugging.
- Tests cover both baseline and anchor-first branches.
- Validation includes overlays, not only metrics.

## Expected Outcome

If the hypothesis is correct, the first good version should keep the useful part of the existing frontend, YOLOWorld semantic anchors plus SAM mask quality, while removing the expensive global anchor/SAM fusion step.

The key decision after validation is simple:

- If accuracy stays near baseline, anchor-first SAM becomes the new default optimization path.
- If accuracy drops moderately, tune multimask/depth/residual settings.
- If accuracy drops like SED++, reject it and optimize the old fusion path directly with vectorized matching.
