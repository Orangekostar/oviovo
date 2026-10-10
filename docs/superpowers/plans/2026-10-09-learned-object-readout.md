# Learned object readout implementation plan

> **For agentic workers:** Use superpowers:executing-plans. The primary agent owns scientific decisions, interfaces, implementation and review. Only explicit deterministic mechanical leaves may use mechanical_worker.

**Goal:** Implement and actually execute the fixed supervised learned-readout study, then publish its verified evidence.

**Architecture:** New task modules adapt verified full-parent inputs without changing parent artifacts. Frozen FP32 FC shards feed four matched trained readouts, whose new F representations update fixed G1 outputs through the original source grouping and official evaluator.

**Tech Stack:** Existing Python mapping and FC environments, NumPy/SciPy, PyTorch/open_clip, Open3D, pinned Mask-Adapter and ScanNet sources.

**Spec:** `docs/paper/static_ovmap/learned_object_readout_v1/CODEX_FINAL_EXECUTION_EN.md` and the three referenced contracts; design in `docs/superpowers/specs/2026-10-09-learned-object-readout-design.md`.

## Global constraints

- Exact JSON constants, 24/4/4 new families, 160/40 classes, 2/4/8 shared prefixes.
- FP32, no autocast/TF32, frozen visual/text parameters, differentiable frozen projection.
- Scientific optimizer updates: seed17 10,000; seed29 6,000. Effective batch16, microbatch1, accumulation16.
- H annotations only after both seed nominations; existing benchmark annotations never select training inputs/checkpoints.
- All nine main and two repeat map paths; 286 logical rows and 22 ordered pools if complete.
- One A40 with lock; data workers4 and evaluation workers3; no cold timing grid, new OVI maps or N/Q inference.
- Preserve all parents. Normal push; external publication proof. Precise nonzero dependency failures.

## Task 1: Verified bindings and frozen splits

Files: `common.py`, `binding.py`, `inventory.py`, public entrypoint, production contract tests.

- [x] Run supplied references: `/home/ww/miniconda3/envs/oviovo-radseg/bin/python -m unittest discover -s docs/paper/static_ovmap/learned_object_readout_v1/reference -p test_reference.py -v`.
- [x] Write failing production split/inventory tests: choose the lowest complete suffix, exclude every prior family, assert 24/4/4 disjoint roles and 160/40 disjoint classes; reject fewer than32 families.
- [x] Implement `family_split(scans, denied)` and `class_split(ids)`, bounded inventory and task-owned full-parent adapter.
- [x] Bind actual sources; record source/dependency/data inventory and fixed metadata. Materialize only the fixed eligible family queue through authorized downloader constants.
- [x] Run targeted production tests; inspect the real split manifest and input identities before annotation-derived generation.

## Task 2: Calibrated proposals and common observations

Files: `sensor.py`, `data.py`, `observations.py` and directed tests.

- [x] Write camera tests with a nonzero depth-to-color translation, color bounds/occlusion and shared axis transform; write exact area-truncation/donor and group-zero mapping tests.
- [x] Port supported sensor decompression, preserve actual calibration and depth_shift, select finite-pose representatives and export registered depth-grid RGB.
- [x] Generate TRAIN/DEV four-condition GT-derived proposals, area/site support and common >=2-view banks with separate annotation sidecars; do not parse H.
- [x] Construct regression manifests from predicted owners with no GT inputs. Validate one actual registration overlay and exact inherited regression camera convention.
- [x] Check actual original-object/category sufficiency; record omissions and no-op corruption counts once.

## Task 3: Frozen features and exact learned heads

Files: `features.py`, `mask_adapter.py`, `model.py`, `losses.py` and directed tests.

- [x] Write feature-center/padding, missing-view, permutation/group-distinction and auxiliary-unknown-mask tests before their kernels.
- [x] Capture actual raw C/projected D and text dimensions; persist ordinary FP32 raw/projected CPU tensors once per image with transform metadata and identities.
- [x] Port pinned MA numerical head and three ConvNeXt blocks, retaining notices. Compare real port outputs/input/parameter gradients against the pinned source with identical weights.
- [x] Implement `ReadoutHead` returning normalized embedding plus training-only local auxiliaries. Preserve raw-pool-before-phi, separate local/global attention and the fixed residual ramp.
- [x] Run real frozen-FC backward and unchanged-weight checks; run/discard the two-TRAIN-object 20-update engineering fit. No hyperparameter selection from this fit.

## Task 4: Exact resumable scientific training and nomination

Files: `training.py`, DEV recognition evaluation, nomination receipts and checkpoint registry.

- [x] Write deterministic draw/optimizer-resume roundtrip test, and DEV equal-family/early-tie selection tests.
- [x] Precompute uniform-class/uniform-object draw schedules shared within each seed, with fixed view/corruption cycles.
- [x] Guard resumed branches against skipped DEV checks. Complete only an owed check at the saved weights; reject unavailable historical weights. Require all four actual DEV receipts before nomination/reporting.19 bounded production tests pass; the numerical training files remain frozen.
- [x] Run seed17 warmup and four branches; validate only500/1000/1500/2000; select each checkpoint by exact DEV criterion.
- [x] Freeze the proposed architecture nomination, run seed29 warmup plus MA and nominated branch, and freeze both seed checkpoint nominations before opening H.
- [x] Verify 16,000 scientific updates, finite losses/gradients, actual parameter movement, and frozen-backbone identities.

## Task 5: Heldout recognition and locked full-map predictions

Files: `prediction.py`, H ingestion, recognition evaluation, owner decision arrays.

- [x] Require both nominations before H annotation ingestion. Generate H with the same frozen data algorithm and check base/novel support without replacing families.
- [x] Evaluate seven main and two separate repeat recognition paths on full200-class text, base/heldout/all subsets and four conditions with original-object denominators.
- [x] Produce all nine main and two repeat paths on 26 real maps; recompute full official current-class ranks. Lock outputs before loading new evaluation labels.
- [x] Run real zero-update whole-output/rank parity and eligibility/protected/recovered invariants; retain standalone F and final-label differences.
- [x] Add a production zero-update whole-output check over real G1 maps, with per-scene proofs and a mandatory26-scene completion gate in `predict`/reporting. All26 scenes completed with whole-output/rank parity proofs.

## Task 6: Official pooled metrics, mechanisms and accuracy gates

Files: `evaluation.py`, `selection.py`, bounded ordered-pool coverage test.

- [x] Implement the new evaluator adapter, exact strict/material gates and fixed-support/trace diagnostics; bounded production tests pass. Actual scientific execution remains below.
- [x] Verify actual evaluator overlap array and minimum100-point rule. Use released ordered pooling, never mean scene AP.
- [x] Execute 234 main/52 repeat records and18/4 pools, allowing only exact scorer-input aliases with identity proofs.
- [x] Compute full5 unrounded metrics, strict/material gates, pre-nominated repeat consistency and exploratory exposed-benchmark winner separately.
- [x] Compute fixed four mechanism contrasts, new/old errors, GT50/75 gained/lost and unchanged class-agnostic matches. Do not alter methods after results.

## Task 7: Reports, terminal CLI and publication audit

Files: `reporting.py`, `runner.py`, four LEARNED_OBJECT reports and compact artifact directory.

- [x] Implement three main table families in MD/CSV/JSON/LaTeX, a separate seed29 supplement, selected-head export and four-report generation.
- [x] Generate exactly three table families in MD/CSV/JSON/LaTeX, with actual denominators, parameter counts, steps, encodes, stage costs and allocated/reserved peaks.
- [x] Preserve all selected small trainable heads and configs, both seed nominations, split/class manifests and real result/decision arrays. Exclude raw scans, GT maps, dense tensors and frozen backbone.
- [x] Run the exact complete CLI with `--phase all --resume` and observe terminal exit0 and SCIENCE_COMPLETE, with the same result identity in the observer and controller receipts. Missing mandatory work returns nonzero with explicit null metrics.
- [x] Review each original contract requirement against current files, execution and artifacts. The finite primary audit passed against all16 original pack files, actual all CLI0,16000 updates,286/22 coverage, all26 output/diagnostic checks and six exact selected head checkpoints; evidence is in primary_review.json.
- [ ] Commit and push normally; compare full local/remote task-branch SHA and write external `publication/final.json` using observed exit/status/result identity. Confirm clean worktree and report actual completion or dependency block.

Actual resource completion: one cache-only replay on all1,436 real-map objects and828 frame inputs; outputs are bitwise identical to the original, zero new encoder calls/optimizer updates. Original missing separate capture/readout clocks remain null. The full public predict invocation separately records CPU payload construction and current-class rank rebuilding.31 bounded tests pass. Full official evaluation, package generation, observed all CLI0 and final primary review are complete; normal publication remains pending above.
