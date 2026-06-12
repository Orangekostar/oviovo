# Metric-Stable Parallel Scheduling Design

## Goal

Speed up the fast high-IoU room0 path without changing the semantic or geometry decision rules. The target accuracy contract is metric-stable acceleration: the 200-frame stride-10 validation should keep `mIoU` and `f-mIoU` within `0.002` of `outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f`, while reducing wall time and exposing long-tail stage behavior.

## Current Context

The current run processes frames sequentially in `run_room0_full_eval.py`, and each frame calls `OVIOVOPipeline.process_frame()` once. Inside a frame, the main stages also execute in order:

1. Load or generate SAM proposals.
2. Generate YOLOWorld and YOLOE supplemental anchors.
3. Build anchor-voted proposals.
4. Run `runtime_vis`.
5. Run `depth_refinement`.
6. Run `patch_lifting`.
7. Derive active set and split background/object patches.
8. Run association.
9. Run object update and TSDF/local-pool mutation.
10. Update semantic memory and dense surface.

Some local parallelism already exists:

- `runtime_vis` can score proposal pairs with a thread pool.
- `patch_lifting` can lift proposals with a thread pool.
- `depth_refinement` supports proposal-level parallelism, but the fast high-IoU config does not enable it.
- YOLOE supplemental anchors use a persistent JSON-lines worker, which avoids per-frame process startup but does not overlap with later frame stages.

The 200-frame baseline had these main stage totals:

- `object_update`: `1334.4s`
- `proposal_generation`: `787.0s`
- `association`: `426.4s`
- `runtime_vis`: `332.5s`
- `depth_refinement`: `107.1s`

The run also had long-tail outliers: one YOLOWorld/proposal frame, one association frame, and two object-update frames dominated wall time.

## Design Choice

Use a conservative ordered-commit scheduler:

- Preserve one-at-a-time map mutation.
- Allow frontend work for the next frame to run while the current frame is doing state-dependent work.
- Add bounded thread-pool parallelism only to pure or read-only computations.
- Keep output ordering deterministic when parallel work is reduced back into pipeline results.

This avoids concurrent writes to `SystemState`, `TSDFInstanceVolume`, `ObjectMap`, or semantic memory. Those structures remain updated only in frame order.

## Components

### Proposal Bundle

Introduce a `FrameProposalBundle` data structure containing all frontend outputs needed by the main pipeline:

- `frame_id`
- `source_proposals`
- `raw_proposals`
- `anchors`
- `anchor_assignments`
- `proposal_source`
- `anchor_guided_sam_summary`
- `generation_timings`
- `actual_backend`

The bundle is immutable after creation from the scheduler's perspective. It does not hold or mutate map state.

### Pipeline Bundle Injection

Extend `OVIOVOPipeline.process_frame()` with an optional `proposal_bundle` parameter. When a bundle is provided, the pipeline skips only the frontend proposal/anchor generation block and fills the same `last_*` fields that the existing path fills:

- `last_raw_proposals`
- `last_source_proposals`
- `last_anchors`
- `last_anchor_assignments`
- stage timings for `proposal_generation`, `yoloworld_primary`, `yoloe_supplemental`, and `anchor_merge`

Every stateful downstream stage stays in the existing order.

### Frame Proposal Prefetcher

Add a small scheduler utility used by `run_room0_full_eval.py`.

Responsibilities:

- Own a single background worker for frontend work.
- Submit at most one future ahead of the currently committed frame.
- Compute `FrameProposalBundle` for frame `N+1` while frame `N` runs downstream stages.
- Return a prefetched bundle if it is ready and matches the expected frame id.
- Fall back to synchronous pipeline frontend if no matching bundle is available.
- Close the worker cleanly at the end of the run.

The first implementation should prefetch only one frame ahead. A deeper queue is intentionally out of scope because it can increase GPU memory pressure and amplify stale work if a run is interrupted.

### Frontend Extraction

Refactor frontend generation out of `OVIOVOPipeline.process_frame()` into a helper method such as `_build_proposal_bundle(frame)`. The helper should contain the existing proposal/object-anchor logic and produce exactly the same proposals that the inline path produced.

`process_frame()` then has two paths:

- no bundle: call `_build_proposal_bundle(frame)`
- bundle provided: validate `bundle.frame_id == frame.frame_id`, then consume it

This keeps one source of truth for frontend semantics and makes prefetch testable without a full pipeline run.

### Association Score Parallelism

Add optional association scoring parallelism behind config flags:

- `association.score_parallel_enabled`
- `association.score_parallel_workers`
- `association.score_parallel_min_candidates`

Only read-only scoring work should run in parallel. The module should still:

- iterate patches in original order,
- compute TSDF vote per patch before parallel candidate scoring,
- reduce candidate scores in sorted `candidate_ids` order,
- apply identity gates and build debug records on the main thread,
- append `matched`, `new_object_patches`, and `contested_object_patches` in the same patch order as before.

This allows speedup in heavy scoring frames while minimizing result drift.

### Depth Refinement Config

Enable existing proposal-level depth refinement parallelism in the fast high-IoU config:

- `depth_refinement.proposal_parallel_enabled: true`
- `depth_refinement.proposal_parallel_workers: 4`
- `depth_refinement.proposal_parallel_min_tasks: 16`

The depth refinement code already maps independent proposals and returns results in input order. The config change should be tested with an equality-oriented unit test and validated in smoke runs.

### Instrumentation

Add run-report and frame-metric fields for scheduling:

- `prefetch_enabled`
- `prefetch_submitted_count`
- `prefetch_hit_count`
- `prefetch_miss_count`
- `prefetch_wait_sec_total`
- `frontend_source`: `inline` or `prefetch`
- `frontend_exception_count`
- `association_score_parallel_used_count`
- `depth_refinement_parallel_used_count`

Keep existing stage timings. Add derived summaries for p50, p95, max, and outlier frame ids for the major stages. The goal is to distinguish steady-state throughput from long-tail failures.

## Data Flow

For frame `N`:

1. The runner obtains frame `N`.
2. The runner retrieves the prefetched proposal bundle for frame `N` if available.
3. Before or immediately after starting processing for frame `N`, the runner submits frame `N+1` frontend prefetch if there is a next frame and no prefetch is pending.
4. The pipeline consumes the bundle or builds frontend inline.
5. The pipeline runs all stateful stages in the original order.
6. The runner records frame metrics, including prefetch source and wait time.
7. The runner advances to frame `N+1`.

The background worker never sees or mutates the current object map. It only uses RGB/depth/frame metadata needed by proposal generation.

## Error Handling

If prefetch raises an exception:

- record the exception type and message in scheduler debug metrics,
- discard the failed future,
- run the frontend inline for that frame,
- continue the run unless the inline path fails.

If a prefetched bundle frame id does not match the expected frame:

- discard it,
- count it as a miss,
- run inline.

If the prefetch worker is slower than downstream stages:

- the runner can wait for the matching future up to a configurable short wait threshold,
- if not ready, it runs inline or blocks depending on config.

The initial config should prefer correctness and simplicity:

- `prefetch_wait_timeout_sec: 0.0` for smoke tests that measure nonblocking behavior,
- allow a later monitored run with a small wait threshold if duplicate frontend work becomes wasteful.

## Testing Strategy

Use TDD for implementation.

Unit tests:

1. Pipeline bundle injection produces the same `last_*` frontend fields as inline generation using test backends.
2. Prefetcher returns matching bundles and rejects mismatched frame ids.
3. Prefetcher falls back cleanly after worker exceptions.
4. Association score parallel path matches serial scores, matches, contested patches, and debug summaries for a deterministic synthetic case.
5. Depth refinement parallel config preserves output order and equivalent proposal results.
6. Config test confirms fast high-IoU scheduling flags are explicitly set.

Validation runs:

1. Focused unit tests for scheduler, pipeline injection, association, runtime vis, depth refinement, and object anchor worker client.
2. A 20-frame stride-10 smoke run with prefetch and parallel scoring enabled.
3. A 200-frame stride-10 room0 run compared to `20260601_room0_fast_high_iou_v2_stride10_200f`.

The 200-frame validation passes if:

- process exits with status `0`,
- `mIoU >= baseline_mIoU - 0.002`,
- `f-mIoU >= baseline_fmIoU - 0.002`,
- no major class audit regression appears in `ceiling`, `wall`, `floor`, `sofa`, `blinds`, `window`, or `lamp`,
- frame metrics contain nonzero prefetch attempts and report the hit/miss counts.

## Non-Goals

- No multi-frame state commits.
- No concurrent writes to TSDF, object maps, semantic memory, or dense surface.
- No detector skipping, confidence threshold changes, class-list changes, or anchor assignment policy changes.
- No change to mIoU calculation or final audit semantics.
- No deep prefetch queue in the first implementation.

## Open Risks

- YOLOWorld appears to use a synchronous helper process. Prefetch can hide some of its cost, but a persistent YOLOWorld worker would be a separate follow-up.
- GPU contention may reduce gains if the prefetch worker runs detector inference while current-frame GPU work is active. The current downstream stages are mostly CPU-heavy, so the first implementation should still be useful.
- Association parallelism uses Python threads. If the bottleneck is Python bytecode rather than NumPy kernels, speedup may be modest. The deterministic test still protects the refactor.
- Object-update long-tail frames are stateful and not safe to parallelize broadly in this iteration. Instrumentation should identify whether later targeted optimization is needed.
