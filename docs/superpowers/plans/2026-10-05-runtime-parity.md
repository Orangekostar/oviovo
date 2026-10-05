# Recovery Runtime Parity Implementation Plan

> **For agentic workers:** Execute inline with `superpowers:executing-plans`. The primary agent owns algorithms, experiments and final review; mechanical leaves alone may use `mechanical_worker`.

**Goal:** Run the complete bounded v2 recovery runtime experiment, preserve every scientific output, publish measured tables and verify the GitHub push.

**Architecture:** Bind the actual local v2 producers and their receipts in an independent commit. A new adapter shares timing/bookkeeping across the unchanged numerical reference and three cumulative execution variants. Verification contexts alone can access parent recovery vectors; cold contexts contain only common geometry, incumbent evidence, model/text and input paths.

**Tech Stack:** Python, NumPy, Open3D CPU raycasting, PyTorch FP32 CUDA, existing released operators and LaTeX.

**Spec:** `docs/paper/static_ovmap/runtime_parity_spec/OVIMAP_RUNTIME_PARITY_CODEX_2026-10-05/CODEX_FINAL_EXECUTION_EN.md`, companion contracts, timing contract and JSON specification.

## Global Constraints

- Branch `research/ovimap-runtime-parity-v1`; independent attempt `/mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001`.
- Actual v2 fallback module and parent science remain immutable; deployment `N0_UNCHANGED`.
- GPU 2, original FP32/model/operator/TF32/determinism settings, image batch 1, raycasting threads 4, ray batches at most 65536.
- Exactly two G1 pilot repeats per variant on room0/office1, at most 16 calls; 3% simplicity selection band.
- At most 52 CPU projection passes; no CF18 neural inference, maps, segmentation, Native crops, fitting or new models.
- Freeze before final: 64 final calls for a selected candidate, or 48 if R0 remains; two repeats, second round reversed.
- Up to four external-failure attempts, once per leaf; never retry slow or parity-failed calls.
- Exact masks, selections, availability, fallback, labels, owners, semantics and rank strings; FP32 evidence atol=rtol=1e-5 with exact argmax.
- Import 172 scene rows and 14 pools once; keep three main tables and supplementary S7/S8; all current times freshly measured.
- At most 25 MiB new compact artifacts; no force push; verify remote full SHA before external publication receipt.

## Task 1: Immutable Source Binding

Files: new `runtime_parity/binding.py`, `reference_binding.json`, `source_to_task.md`, and task config.

- [x] Inspect actual source HEAD/status, producers, v2 receipt lineage and fixed cohort.
- [x] Create the specified worktree; copy only measured v2 producers, tests and compact reports.
- [x] Commit the reference snapshot; bind producer/operator/model hashes, effective pooling math, all 26 scenes and U2 legacy semantics.
- [x] Verify all 172 row receipts and 14 pool receipts against the v2 result store once and retain original identities.

## Task 2: Exact Projection And Input Reuse

Files: new `runtime_parity/kernels.py`, `views.py`, focused `tests/evaluation/test_runtime_parity.py`.

Interfaces: `PreparedCamera.rays(pixels)`, `ExactProjector.project_frame(K, pose, depth, candidates)`, `retain_top3(values, pixels, frame_id, materialize)`, and `build_views(inputs, root, implementation, index, stages)`.

- [x] Write and observe failing production tests for ray bytes, full-scene occluders, mixed owners, near-plane fallback, depth tolerance, grouped counts and exact Top-3 ties.
- [x] Keep original `FullSceneProjector` for R0. R1 groups owner statistics and validates/inverts each camera once. R2 queries conservative union rectangles against the same complete BVH.
- [x] Preserve original frustum diagnostic values separately from conservative ROI logic using the true inverse rotation and FP32 ray-origin rounding.
- [x] Parse manifests/capture once per call and decode one RGB per physical content key in R1+; keep exact mask and camera checks.
- [x] Run focused tests with `PYTHONPATH=src /home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest tests/evaluation/test_runtime_parity.py -q`.

## Task 3: Cold Adapter, Instrumentation And Parity

Files: new `runtime_parity/session.py`, `runner.py`, `parity.py`, `workflow.py`, CLI.

Interfaces: `recover(inputs, arm, implementation, model_session, run_context)` and independent cold/verification context dataclasses; exclusive `Stages.span(name)`.

- [x] Test production v2 normal/empty fallback, scientific content key bijections, cold context isolation, two-repeat identity/ordering and table units.
- [x] Load common geometry/NQF once; execute original preprocessing/pooling/head/classification/export. R3 defers already-computed vector transfer per frame and releases every live scene tensor without per-image allocator clearing.
- [x] Add synchronized total wall timer, exclusive host stages, non-additive CUDA service events, actual counters and memory peaks. Keep every output write inside the common return boundary.
- [x] Implement all requested phases and source-aware resumable ledger; check authoritative live process identity before retry.
- [x] Record one shape-only warm-up per model process and its independent GPU cost.

## Task 4: Execute Bounded Evidence

- [x] Run profile/pilot slots in fixed ascending/reverse order; compare every call against parent scientific evidence after the timer.
- [x] Select by two-scene/two-repeat mean and the 3% simplicity rule, including R0.
- [x] Verify all 26 candidate per-frame statistics, ordered Top-3 and mask bytes, using parent vectors only for non-timed CF18 export parity.
- [x] Commit implementation freeze, hardware/settings, selected flags and exact final call order.
- [x] Run every required final Replica cold call, including contemporary U2; retain outliers and original failure costs.

Execution command: `/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_runtime_parity.py --spec configs/static_ovmap/runtime_parity_v1.json --source-worktree /mnt/shared/ww/ovimap-cvpr-compact-tables-v1/worktree --v2-results-root /home/ww/ovimap-area-fallback-v2/attempt_001 --phase all --resume`.

## Task 5: Tables, Requirement Audit And Publication

Files: `runtime_parity/reporting.py`, four required `RUNTIME_PARITY_*.md` reports and compact artifacts.

- [x] Preserve Table1/2 science; render Table3 and S7/S8 from machine data with per-cell source identities and all sixteen constituent final time IDs.
- [x] Publish per-scene/repeat CSVs, stage/service times, counters, memory and source/parity maps; render and visually inspect PDFs.
- [x] Audit every numbered requirement, contract and named deliverable against actual artifacts; report distinct source, implementation, parity, timing, metric-reuse, speed and publication statuses.
- [x] Verify original source HEAD/index unchanged and compact artifact size below 25 MiB.
- [x] Commit deliverables, run ordinary push, compare local full SHA to remote branch, then write external `publication/final.json`.

## Acceptance Evidence

Binding: measured producer hashes and immutable reference commit. Projection: production fixtures plus 26 scene receipts. Neural/export: all pilot/final discrete and floating comparisons. Timing: complete source-bound two-repeat ledger and positive stage totals. Metrics: imported original 172/14 identities plus export parity. Publication: rendered artifacts, requirement audit and exact remote SHA.
