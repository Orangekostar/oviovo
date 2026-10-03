# CVPR Compact Tables Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. The primary agent owns
> semantics, algorithms, implementation, debugging and final review. Delegation
> is limited to fully specified mechanical leaves under the user's AGENTS.md.

**Goal:** Execute the fixed 26-capture, eight-condition protocol and publish three
measured compact tables with complete provenance and actual cold timing.

**Architecture:** A new task-local pipeline binds immutable parent artifacts and
prepares only missing fixed-cohort assets. Projected recovery and unknown-aware
outputs use new interfaces while retaining pinned native/FC/scoring operators.
Typed complete-cohort results supply every table occurrence.

**Tech Stack:** Existing ovimap-map/oviovo-radseg/CropFormer environments, NumPy,
Open3D CPU RaycastingScene, pinned FP32 SigLIP and FC, released scorer, LaTeX.

**Spec:** `docs/superpowers/specs/2026-10-03-cvpr-compact-tables-design.md` and the
unchanged imported execution package.

## Global Constraints

- Base 1a1c4513ecf824c55274a456c40657271d631883; upstream f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424.
- Branch research/ovimap-cvpr-compact-tables-v1; output /mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001.
- Replica8 and CF18 exact order; 26 common anchors, 172 outputs, 14 pools, 24 timing replays.
- BB00_NATIVE only, voxel .01m, association 4, original 200-slot schedules without refill.
- Fixed final Replica N/Q/F temperatures shared by both cohorts; no new fits.
- Candidate residual >=100 source rows, cap128; full-mesh 4-thread raycast batches <=65536.
- Camera-z rays, fixed max(.02,.02*depth) tolerance, >=100 pixels, bbox extents >=2.
- G1 prefix of preselected G3; no replacement of failed views; no historical-request gating.
- Fixed A3 primary, no performance gate, no automatic deployment, no cross-source SOTA claim.
- Freeze complete implementation before new main predictions/metrics; synthetic tests and one existing scene0056_00 smoke only.
- Existing environments; no external model installs, new mapper variants, additional benchmarks or whole-repo tests.
- New caches writable, parents read-only; two map workers x8 threads, three eval workers x4 BLAS, one GPU worker.

## Task 1: Binding And Fixed Acquisition

Files: `src/static_ovmap/cvpr_compact/{protocol,binding,benchmark_inputs}.py`,
`scripts/evaluation/run_ovimap_cvpr_compact.py`, new focused tests.
Consumes parent resolved_inputs, map/readout/capture/model/producer receipts.
Produces immutable task binding, availability ledger and exact fixed CF18 export receipts.

- [x] Inspect frozen package and actual repository/parent state; create required isolated worktree.
- [x] Import package unchanged and copy executable JSON verbatim.
- [x] Add tests for exact 26/172/14/24 bookkeeping and final-temperature equality; verify actual parent identities during binding.
- [x] Implement a bounded one-time data-root search, full filesystem-stamp memo, repeated path maps and fixed-list download/export.
- [x] Bind effective archived producer sources, actual model/text/Q-checkpoint identities, commands and exposure lists.
- [x] Run bind and prepare with the existing authorized downloader; all 18 fixed captures completed with original native schedules.

## Task 2: Independent Projected Requests

Files: `src/static_ovmap/cvpr_compact/projected_views.py` and focused tests.
Interface: `FullSceneProjector(xyz, faces, raw).project_frame(K, pose, depth, candidates)`
returns owner masks and per-stage counts; `build_projected_views(...)` returns
new requests, geometry-only G1/G3 selection and receipt.

- [x] Write fixtures for oblique camera-z rays, front owner-zero occlusion, invalid depth, degenerate primitive mapping and exact batch parity.
- [x] Implement full-scene BVH, frame validation and the fixed depth predicate.
- [x] Write tests proving a visible candidate with no legacy request receives G1, distinct-frame G3 has the exact G1 prefix, bbox conventions and mask readback agree.
- [x] Implement new projected schema, compact selected masks and truthful camera/image/depth/source identities.
- [x] Verify on synthetic inputs, then inspect a GT-free scene0056_00 projection montage for camera alignment.

## Task 3: Anchors, Readouts And Output Contracts

Files: `src/static_ovmap/cvpr_compact/{anchor,base_sources,region_worker,native_region_worker,outputs,prediction_worker}.py`.
Consumes bound inputs, one BB00 capture and locked projected/legacy requests.
Produces N/Q/F source vectors, selected-view evidence and immutable A0-A5/U2/G3 payloads.

- [x] Implement CF18 command substitutions with the actual parent BB00 binary/config, original CropFormer PNG conversion and native capture hooks; scene0011_00 completed200/200 slots with the exact bound kernel.
- [ ] Reuse compatible maps; build missing maps once; reproduce original N and causal Q with honest exhaustion, static F <=3 views and fixed final temperatures.
- [x] Implement task-local N/Q/static-F bridges; test separate N cosine, original F selection/cap exclusions, committed-freeze-first entry gates and successful-worker reuse without spending retries.
- [x] Implement FC frame sharing with exact image/model/operator identity; pool new masks with inherited signed-mask operators.
- [x] Implement same-view native six-crop recovery with original canonical-relative FP32 classifier.
- [x] Test A5 unknown0 preservation and zero rank, residual-only append, no incumbent mask replacement and common A2/A3/A5 recovered set.
- [x] Produce eight real smoke outputs on existing scene0056_00 and verify exact A0 arrays/ranks; no new development map or main-scene scoring.
- [x] Implement method-specific Native/Q/F/recovery content unions, A5 Native support prerequisites, once-per-worker physical payments and failure retention; verify all8 existing Replica and original smoke inventories without neural inference.

## Task 4: Freeze, Official Evaluation And Diagnostics

Files: `src/static_ovmap/cvpr_compact/{workflow,evaluation,diagnostics}.py`.
Consumes locked outputs; produces 172 typed rows, 14 pools and actual matching evidence.

- [ ] Commit complete implementation plus freeze/experiment.json and exposure ledger before main predictions.
- [ ] Expand mask registry with genuine Native-unavailable recovered owners, preserve unknown errors and exact original 1NN projection.
- [x] Implement the expanded-registry scoring wrapper and exercise the actual released evaluator on a synthetic full-owner fixture, including A5 unknown errors and exact resumption.
- [x] Implement bounded workflow/CLI, shared recovery export and original per-class scene/pool summaries; verify original AP/semantic parity and successful pool resumption on synthetic data.
- [x] Check actual released nine overlaps within1e-12 and retain original vector; test exact alias contexts including arrays, ranks, GT bytes and actual official view.
- [ ] Execute all missing main leaves; lock predictions before label diagnostics; pool only full ordered cohorts.
- [ ] Derive common n/N, target-min100 coverage, added matcher TP/FP entries, ambiguity flags, old lost matches and changed ranks.
- [x] Implement post-lock diagnosis using the unchanged original matcher observer; test sum-of-count n/N, actual TP50 additions, lost old matches and omitted unknown-owner ranks.

## Task 5: Real Cold Timing

Files: `src/static_ovmap/cvpr_compact/timing.py` and the shared production recovery callable.
Produces exactly24 synchronized resident-model, feature-cache-cold receipts.

- [x] Implement one shared production callable, fresh cold views/cache-disabled sessions, exclusive nonreplayable reservations, exact labels/support plus declared FP32 parity, full-arm-name mapping, and all-eight-scene means; synthetic tests pass.
- [ ] Isolate empty feature/view/result caches per scene/arm; forbid parent persistent feature reads, permit resident model/text and within-run frame sharing only.
- [ ] Measure candidate creation, real view/archive work, decode, encode, pool, classify and output/rank export under one timer; record model load separately.
- [ ] Verify scientific support/label and FP32 feature parity; average all8 scene costs per arm, including real no-op eligibility overhead.

## Task 6: Typed Tables And Claims

Files: `src/static_ovmap/cvpr_compact/tables.py`, four COMPACT_TABLES reports,
`artifacts/static_ovmap/cvpr_compact_tables_v1/` compact deliverables.

- [x] Verify arXiv-v1 and final-CVF PDFs; all30 values agree. Preserve the unchanged provided transcription and its7 differences, explicitly choose the verified PDF representation, and retain external protocol/precision/supervision cautions.
- [x] Implement one complete fractional scene/pool store and typed cells with receipt identities or source coordinates; test nulls, fixed full coverage, exact order and repeated A3 identities.
- [x] Implement6/6/4-row booktabs TeX, JSON, CSV and cell provenance from typed data; compile and visually inspect a clearly marked synthetic9pt preview without overflow.
- [ ] Populate the final generated files from actual172 outputs,14 pools and24 cold calls; synthetic artifacts are not release results.
- [ ] Compile actual preview at >=8.5pt, visually inspect PDF and overflow; preserve all fixed rows and negative results.
- [ ] Report full five metrics, paired A3-A0/A1/A4/A5 deltas, 2x2 effects, Pareto/guardrails, actual coverage/cost and separate status fields.
- [x] Implement four report generators and paired analysis; test fixed A3, negative A3-A5 effects, null deltas, Pareto controls and conditional time differences. Actual reports remain pending full measured evidence.

## Task 7: Acceptance And Publication

- [x] Run the final current-source focused production validation (51 tests) and real scene0056_00 eight-condition end-to-end check with exact baseline and shared-export parity; no whole-repository test sweep.
- [ ] Execute the specified all --resume command and verify idempotent exact-input reuse.
- [ ] Audit every package requirement against actual files, receipts and outputs; retain unresolved items and keep goal active until full completion.
- [ ] Keep compact release below50MiB, large maps/weights on shared storage with truthful manifests.
- [ ] Commit and normal-push the named branch; compare full local/remote SHA and write external publication/final.json without another recursive verification commit.

Canonical execution:
```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_cvpr_compact.py --spec configs/static_ovmap/cvpr_compact_tables_v1.json --phase all --resume
```

## Current Execution Evidence

- Existing main native anchors:8 exact Replica aliases. New main native anchors:18 complete CF18 captures, each200/200 original frames (26/26 total).
- All18 CF18 raw input/export and CropFormer receipts are complete. The original controller in `attempt_001/execution/anchors/background_controller.json` completed; its kernel PID is absent and all26 actual anchor contexts were checked.
- Final focused contract suite:51 passed, including synthetic integration through the pinned scorer, captured original per-class pools, cold-source parity, all-failed technical blocking, table provenance, negative analysis, content unions, paid-failure provenance, orphan locks, exact retry bounds, complete compact packaging and lossless legacy storage conversion. Compilation and scoped whitespace checks passed. This is not new main-benchmark scoring or another development map.
- Existing Native/Q/F cost inventories for8 Replica scenes and original smoke are checked, including failed dense inputs recovered from the original paid array ledgers. The full main physical ledger remains pending actual execution.
- Main outputs0/172, internal pools0/14, actual cold timing0/24. Freeze/publication modules and live/orphan guards are implemented. The current original eight-condition smoke passed, including exact warm/final export and A0 parity. The freeze commit, full measured execution and final release remain pending.
- Previous scene0056_00 smoke receipts and failed commands are preserved. Explicit pre-freeze regeneration corrected only lossless legacy FP32 storage and environment-dependent probability-audit metadata; no scientific parameter, label decision, extra method or development map changed.
