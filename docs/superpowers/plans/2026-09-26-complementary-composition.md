# Complementary composition implementation plan

> **For agentic workers:** Execute task-by-task with superpowers:executing-plans. Primary agent owns algorithms, binding, experiments and review. Delegate only explicitly specified mechanical leaves to mechanical_worker (gpt-5.6-luna, max).

**Goal:** Implement and actually measure the six mandatory compositions, CAL-gated M6, frozen nomination, bounded confirmation, and verified publication.

**Architecture:** A new `src/static_ovmap/composition_study/` package reads immutable native-v10 assets and writes only new attempt roots. Prediction builders never consume GT. Native query controllers and forced SigLIP2 readout use separate stores and states; evaluation, scalar fitting, nomination and confirmation are explicit downstream boundaries.

**Tech Stack:** Existing isolated Python environments, NumPy, SciPy, Torch FP32, pinned released ScanNet200 evaluator and native-v10 extension. No new dependency/model installation.

**Spec:** `docs/paper/static_ovmap/complementary_composition_v1/spec/CODEX_FINAL_EXECUTION_EN.md` and `PROTOCOL_SPEC.json` (copied unchanged from supplied ZIP).

## Global constraints

- Base `498f2c5a2fa511c919a83ff9c27946e37cb46a0f`; task branch `research/ovimap-complementary-composition-v1`; sibling worktree `/home/ww/crove/ovimap-complementary-composition`.
- CAL `scene0056_00, scene0534_00`; regression `scene0445_00, scene0626_00`; confirmation `scene0553_00, scene0064_00` only after new lock and exposure check. Historical FIT read-only.
- Preserve exact N0 geometry, owners, projection, vocabulary and serialized ranks; M1/M2 branch S, M3–M6 branch Q.
- B200; one barrier/frame; no refunds or resurrected drops; M5 one state/budget, odd COMBINE/even GAIN global attempted slots with work-conserving fallback.
- Source GPU lock `/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-2.lock`; one model resident; 8 CPU threads. Do not remap development scenes, retrain heads, download models/data or change old locks.
- Three source temperatures only, bounds `[0.01,2.0]`, default `.07`, minimum 5 objects/2 classes, log-space bounded minimization with xatol `1e-4`, maxiter `64`.
- Six mandatory methods × four development scenes = 24 new records; CAL-gated M6 adds four. No regression-driven settings or nominee changes.
- New-only compact artifacts and four evidence tables; code commit A frozen before regression/confirmation, release commit B in external publication receipt; ordinary push and full remote SHA comparison.

## Task 1 — bind immutable inputs and executable attempt lifecycle

Files: `composition_study/binding.py`, `io.py`, public `scripts/evaluation/run_ovimap_composition_study.py`, `configs/evaluation/ovimap_composition_study.json`; tests `tests/composition_study/test_binding.py`.

- [ ] Read source configs, receipts and actual dependency identities; construct one path/stat-keyed source index. Check confirmation metadata without opening images/labels. Missing sources get per-method statuses.
- [ ] Test changed source content invalidates reuse; incompatible settings create a distinct attempt; confirmation role cannot load predictions before lock.
- [ ] Implement `bind(spec_path, output_root, overrides=None) -> resolved_config_path` and `SourceIndex.identity(path, expected=None) -> {path,bytes,sha256}`; phase outputs carry content keys and immutable receipts.
- [ ] Bind checkpoint AND scaler from recorded calibration checkpoint, native/S2 text vocabularies and processors, evaluator and capture inputs. Reuse reader APIs only, never old high-level study execution.

## Task 2 — recover real all-owner source evidence and direct fusion

Files: `object_evidence.py`, `label_fusion.py`; tests `test_evidence_fusion.py`.

- [ ] Define source bundle with scene/native identity, valid_ids/text identity and every owner: label, availability, fallback reason, used/retained request IDs, complete scores or null, readout/source receipts.
- [ ] Test fallback S2 agreeing with Q cannot overwrite N0; one missing source renormalizes; noncontiguous IDs and exact ties retain official order; mismatched geometry/vocabulary fails.
- [ ] Implement `fuse_labels(native_labels, sources, valid_ids, method, temperatures=None) -> (labels, decisions)` with M1 and arithmetic probability M2. No GT arguments.
- [ ] Reconstruct N0 FP32 canonical-relative scores from exact saved native aggregation, static S2 raw-mean top3 AREA and query post-reconciliation scores. Assert reconstructed argmax equals source prediction for every available owner.

## Task 3 — paid native controllers, fixed replay and mixed policy

Files: `visual_requests.py`, `trajectory_reread.py`, `mixed_query.py`; tests `test_trajectories.py`.

- [ ] Test full attempted sequence, model-separated stores, failed/dropped evidence, one pre-batch ranking and debit barrier, unique mixed requests and global alternating lane preference.
- [ ] Implement `replay_fixed(frames, decisions, store, text) -> result` using existing QueryPolicyState/CurrentLineage and final numerical reconciliation. Preserve all native attempted requests and mask/image identities.
- [ ] Implement `replay_mixed(frames, store, text, predictor, standardizer, budget=200) -> result`; call combine admission once per valid frame including quota zero, rank both lanes once, debit batch, update shared overlap history once per success.
- [ ] Reuse verified native/S2 features only after canonical model/processor/dtype/request identity reconciliation; new work writes new model-separated cache. Physical ledger distinct from conservative additive required method operations.
- [ ] Run one real CAL native fixed replay; labels AND retained IDs must exactly match old Q_COMBINE. Reuse this result. Generate missing CAL Q_GAIN and new M5 without refitting Q.

## Task 4 — cross-fitted temperatures and CAL-only nomination

Files: `temperature.py`, `selection.py`; tests `test_temperature_selection.py`.

- [ ] Test unsupported source uses .07, split isolation, finite positive temperatures; M6 only opens under specified mean weak dominance; exact N0/no-intervention cannot win nomination; regression rows rejected.
- [ ] After locking sources, use semantic_events for strict unique-IoU calibration examples. `fit_temperature(examples, training_scenes)` minimizes scene-balanced NLL; keep all inference owners regardless of target availability.
- [ ] Measure six CAL configurations and four controls, evaluate M6 gate, then conditional M6. Nominate by CAL uAP/mIoU/logical count/fixed order with `1e-10` tolerance.
- [ ] Refit temperatures once on both CAL scenes after nomination. Commit implementation A; freeze immutable selection with actual code hashes, model/input identities, final T, nominee, best single, gate and exact confirmation method set.

## Task 5 — regression and bounded new confirmation

Files: `execution.py`, `capture_bridge.py`; tests `test_execution_confirmation.py`.

- [ ] Test exact scene/method/settings contract enforcement, old guard unchanged, no label export before freeze, and exposure blocks both confirmation scenes.
- [ ] Run every mandatory regression configuration, M6 iff CAL gate; final frozen temperatures; no re-selection.
- [ ] Implement capture bridge reusing exact sensor/annotation/frontend/native per-scene recipe and preselected schedules, new exports/captures only, at most two. Preserve original confirmation role provenance.
- [ ] Build actual held-out N0/native ranks/projection; run nominee and four controls (plus M5 only for M6 nominee). Explicit exposure/asset/no-intervention block is separate from runnable CAL/regression.

## Task 6 — evaluation identity, diagnostics and four-table reporting

Files: `evaluation.py`, `reporting.py`; tests `test_reporting.py`.

- [ ] Test persistent cache key contains full prediction/evaluator/projection/GT/vocabulary content; equal predictions can share evaluations but retain distinct records. Never GT before locked prediction.
- [ ] Run actual evaluator and save metrics/trace references. Build Table A per-scene/all means with denominators and costs; B all-owner availability/correctness/routing; C 2×2 and mixed mechanics; D frozen nomination/confirmation/worst-scene and major released gains/losses.
- [ ] Verify one real leaf→released evaluator→report path and resume returning nonempty recorded rows. Explain unchanged AP with changed labels using real eligibility/matches.

## Task 7 — public all-phase execution and bounded verification

- [ ] Public runner implements bind, prepare-cal, calibrate, compose-cal, freeze, regression, confirm, report, publish, all; leaf --resolved-config; actual executable jobs and truthful exit status.
- [ ] Run focused new tests, scoped lint/compile; rerun only affected tests after defects. No old 199-test suite, full old release audit or whole-repository dynamic tests.
- [ ] Run real ordered pipeline; preserve failed identities and label repaired outputs. Check actual operation caps (native <=2400, trajectory S2 <=2400, static S2 <=1536 for CAL/regression).

## Task 8 — requirement-by-requirement final review and publication

- [ ] Personally map all 13 contract sections and JSON constraints to actual files, metrics, commands, gates and traces; unresolved/indirect evidence remains incomplete.
- [ ] Commit scoped code/config/spec/tests, compact new numerical artifacts, scalar fits, selection snapshot, reconstruction commands and `COMPOSITION_RESULTS.md`/`COMPOSITION_HANDOFF.md`. External content manifest for large arrays/models, no copy of old release.
- [ ] Release commit B, ordinary push to task branch, compare full local HEAD with remote. Write external `publication_receipt.json` after push.
- [ ] Final response: measured composition table, nomination/CAL/confirmation reasoning, inference/crop/cache counts, separate implementation/science/confirmation/publication statuses, report paths and verified SHA.

## Current evidence

2026-09-26: source worktree remains untouched at the pinned revision. New attempt_001 binds 97 actual source assets and records a metadata-only confirmation exposure check. Both CAL N0 and static S2 source score tables reproduce every frozen top1 label; all 1,684 saved native observations across the two scene pickles have unique paid frame/bbox/area/pose matches.

The required real scene0056_00 native forced replay passed exact label and retained-ID parity: 200 logical attempts, 1,200 logical crops, 200 cache hits, zero visual forwards. Its outputs were reused in the formal Q_COMBINE leaf. Five actual released-evaluator control records now exist, with exact trace parity. The public leaf→evaluator→report path and a nonempty cache-resume path were exercised. The report explicitly remains PARTIAL: no mandatory composition row, nomination, regression or confirmation is yet complete.

The new public runner, source evidence, temperature jobs, ordered pipeline, confirmation bridge, four-table reporter, reconstruction and ordinary publication implementation are present. The primary agent inspected the mechanically copied native capture recipe and retained review ownership. Twenty-eight focused new tests passed; scoped Ruff, compilation and diff checks passed. No old suite or whole old release audit was run.

Runtime blocker observed: the configured GPU 2 is occupied by an unrelated persist4d project. The real Q_GAIN leaf stopped at the existing GPU occupancy guard before any new inference. The other process was not changed and no device/model/precision fallback was introduced. Continue CPU-side verification and resume the actual visual leaves when the bound GPU becomes available. The goal is not complete.
