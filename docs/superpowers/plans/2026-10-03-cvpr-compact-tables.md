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
- [x] Reuse compatible maps; build missing maps once; reproduce original N and causal Q with honest exhaustion, static F <=3 views and fixed final temperatures. All 18 actual N/Q/F sources were audited; all 18 Q runs made 200 unique acquisitions.
- [x] Implement task-local N/Q/static-F bridges; test separate N cosine, original F selection/cap exclusions, committed-freeze-first entry gates and successful-worker reuse without spending retries.
- [x] Implement FC frame sharing with exact image/model/operator identity; pool new masks with inherited signed-mask operators.
- [x] Implement same-view native six-crop recovery with original canonical-relative FP32 classifier.
- [x] Test A5 unknown0 preservation and zero rank, residual-only append, no incumbent mask replacement and common A2/A3/A5 recovered set.
- [x] Produce eight real smoke outputs on existing scene0056_00 and verify exact A0 arrays/ranks; no new development map or main-scene scoring.
- [x] Implement method-specific Native/Q/F/recovery content unions, A5 Native support prerequisites, once-per-worker physical payments and failure retention; verify all8 existing Replica and original smoke inventories without neural inference.

## Task 4: Freeze, Official Evaluation And Diagnostics

Files: `src/static_ovmap/cvpr_compact/{workflow,evaluation,diagnostics}.py`.
Consumes locked outputs; produces 172 typed rows, 14 pools and actual matching evidence.

- [x] Commit complete implementation plus freeze/experiment.json and exposure ledger before main predictions. Initial freeze `1373d7a5` and documented alias amendment `fd77ab22` are retained.
- [x] Expand mask registry with genuine Native-unavailable recovered owners, preserve unknown errors and exact original 1NN projection for all available actual conditions.
- [x] Implement the expanded-registry scoring wrapper and exercise the actual released evaluator on a synthetic full-owner fixture, including A5 unknown errors and exact resumption.
- [x] Implement bounded workflow/CLI, shared recovery export and original per-class scene/pool summaries; verify original AP/semantic parity and successful pool resumption on synthetic data.
- [x] Check actual released nine overlaps within1e-12 and retain original vector; test exact alias contexts including arrays, ranks, GT bytes and actual official view.
- [ ] Execute all missing main leaves; lock predictions before label diagnostics; pool only full ordered cohorts.
- [x] Execute office1's four independent fixed conditions and four full Replica pools through canonical phases, retaining its four proved blocked conditions and every prior successful record.
- [ ] Derive common n/N, target-min100 coverage, added matcher TP/FP entries, ambiguity flags, old lost matches and changed ranks.
- [x] Implement post-lock diagnosis using the unchanged original matcher observer; test sum-of-count n/N, actual TP50 additions, lost old matches and omitted unknown-owner ranks.

## Task 5: Real Cold Timing

Files: `src/static_ovmap/cvpr_compact/timing.py` and the shared production recovery callable.
Produces exactly24 synchronized resident-model, feature-cache-cold receipts.

- [x] Implement one shared production callable, fresh cold views/cache-disabled sessions, exclusive nonreplayable reservations, exact labels/support plus declared FP32 parity, full-arm-name mapping, and all-eight-scene means; synthetic tests pass.
- [x] Implement per-arm preflight for all24 fixed leaves, actual independent office1 U2 evidence, proved unmeasured G1/G3 blocks, and 22-call single-model/resumption regression checks; freeze amendment03 before physical measurement.
- [x] Isolate empty feature/view/result caches per actual scene/arm call; forbid parent persistent feature reads, permit resident model/text and within-run frame sharing only. All22 valid calls verified; two proved office1 blocks have no reserved call.
- [x] Measure candidate creation, real view/archive work, decode, encode, pool, classify and output/rank export under one timer; record model load separately. All22 valid observations completed on the bound A40, with one model load.
- [ ] Verify scientific support/label and FP32 feature parity; average all8 scene costs per arm, including real no-op eligibility overhead.
- [x] Independently verify all22 actual feature/support/label/rank parities and positive empty-U2 overhead; preserve all24 fixed positions, null blocked G1/G3 means and exact canonical resumption without replay.

## Task 6: Typed Tables And Claims

Files: `src/static_ovmap/cvpr_compact/tables.py`, four COMPACT_TABLES reports,
`artifacts/static_ovmap/cvpr_compact_tables_v1/` compact deliverables.

- [x] Verify arXiv-v1 and final-CVF PDFs; all30 values agree. Preserve the unchanged provided transcription and its7 differences, explicitly choose the verified PDF representation, and retain external protocol/precision/supervision cautions.
- [x] Implement one complete fractional scene/pool store and typed cells with receipt identities or source coordinates; test nulls, fixed full coverage, exact order and repeated A3 identities.
- [x] Implement6/6/4-row booktabs TeX, JSON, CSV and cell provenance from typed data; compile and visually inspect a clearly marked synthetic9pt preview without overflow.
- [ ] Populate the final generated files from actual172 outputs,14 pools and24 cold calls; synthetic artifacts are not release results.
- [x] Compile actual preview at >=8.5pt, visually inspect PDF and overflow; preserve all fixed rows and negative results. Actual one-page9pt PDF has6/6/4 rows and no overfull boxes; primary visual QA is sealed separately.
- [ ] Report full five metrics, paired A3-A0/A1/A4/A5 deltas, 2x2 effects, Pareto/guardrails, actual coverage/cost and separate status fields.
- [x] Implement four report generators and paired analysis; test fixed A3, negative A3-A5 effects, null deltas, Pareto controls and conditional time differences. Actual reports remain pending full measured evidence.

## Task 7: Acceptance And Publication

- [x] Run the current-source focused production validation (66 tests after amendment03) and real scene0056_00 eight-condition end-to-end check with unchanged lock identity; no whole-repository test sweep.
- [ ] Execute the specified all --resume command and verify idempotent exact-input reuse.
- [x] Audit all327 package requirements against actual files, receipts and outputs. The primary review retains28 scientific-blocked requirements and7 subsequent publication obligations; objective_complete=false and the full goal remains active.
- [x] Keep compact release below50MiB, including the primary review. The verified amendment08 artifact set plus review occupies44,588,721 bytes; all1,638 archived members and1,658 original source copies retain exact bytes/hashes. Large immutable dependencies remain in shared storage with reconstruction identities.
- [ ] Commit and normal-push the named branch; compare full local/remote SHA and write external publication/final.json without another recursive verification commit.

Canonical execution:
```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_cvpr_compact.py --spec configs/static_ovmap/cvpr_compact_tables_v1.json --phase all --resume
```

## Current Execution Evidence

- Existing main native anchors:8 exact Replica aliases. New main native anchors:18 complete CF18 captures, each200/200 original frames (26/26 total).
- All18 CF18 raw input/export and CropFormer receipts are complete. The original controller in `attempt_001/execution/anchors/background_controller.json` completed; its kernel PID is absent and all26 actual anchor contexts were checked.
- Final focused contract suite:56 passed under amendment01, including synthetic integration through the pinned scorer, captured original per-class pools, cold-source parity, all-failed technical blocking, table provenance, negative analysis, content unions, paid-failure provenance, orphan locks, exact retry bounds, complete compact packaging, lossless legacy storage conversion and actual current-alias interpretation. The original eight-condition development smoke remains byte-identical.
- Existing Native/Q/F cost inventories for8 Replica scenes and original smoke are checked, including failed dense inputs recovered from the original paid array ledgers. The full main physical ledger remains pending actual execution.
- Freeze revision `fd77ab2203f7e8af2ba9cd9e2ef36d652713ad24` is committed. All 26 projected-view receipts are complete; the primary audited every source registry, completed camera and 454 selected masks. All 18 actual FC weight/class-order audits and Native/Q audits are complete.
- The canonical controller `383645` is absent and its encode phase is FAILED. Its office0 warm recovery completed; the actual failed scene is office1. All three exact preselected FC masks disappear at the original 24x42 dense resolution. The CPU reproduction and three failed command receipts are retained in `validation/office1_dense_support_root_cause.json`; this is a C5 technical block, not a valid zero-recovery output. No mask dilation, alternate view, changed resolution or new scientific recipe is permitted.
- Unchanged frozen production workers completed the other 24 scenes through `validation/continue_independent_encode.py`. A separate CPU controller used at most 3 workers to lock and score ready scenes. Both controllers are terminal, with exact PID/start-tick/argv/cwd history and terminal receipts under `execution/independent_*`. Office1's independent Native and genuine no-archived-view U2 leaves also completed; the original FC failure remains unchanged.
- Actual main predictions and scoring rows are 164/172 across 25 fully completed scenes. All 6 CF18 pools cover their complete ordered 18-scene cohort; no Replica subset was pooled. The primary integration evidence is `validation/independent_execution_integration.json`, identity `ba72c1ee0b5be4e3b35ea7a605463b912de6915caa13a92a7b9074236f405db6`.
- The primary audited all 25 complete FC recovery receipts against the actual selected plans, FP32 vectors, original FC text, area-weighted classification, class IDs and recorded failed views. Evidence is `validation/primary_warm_fc_recovery_audit.json`, identity `9b7786f495c4797e254dad037750d9582610d2e563dc545fa81e09821f1d10ab`. Office1 is explicitly excluded as the retained block, not counted as a complete recovery.
- At the pre-amendment02 stage no actual cold timing had been reserved or measured. Subsequent evidence below supersedes that status; incomplete cohort results must not be reported as official pools.
- Amendment02 is committed at `48fb39d3beae0a50ad369fc72c4bd439fca3fa7f`; current-source final validation passed63 tests and the original eight-condition smoke reused the exact `ccf0b2fd2cf29376abdb426abbdd9c46800ea2dbb07aa1ba786561629d568774` lock. Eight targeted checks cover full ordered pools, exact per-class aliases and no retry of an exhausted FC leaf.
- Canonical encode, predict and evaluate phases are terminal and explicitly `PARTIAL_WITH_TECHNICAL_BLOCKS`. Actual coverage is168/172 outputs and10/14 whole-cohort pools. Office1 A0/A1/A4/U2 are measured; A2/A3/A5/G3 remain blocked without zero predictions or subset pools. The primary verified every completed payload/scoring hash, full ordered pool input and per-class confusion, A0/A1/U2 parity, A4 append-only support and unchanged prior164/six CF18 pools. Integration proof is `validation/partial_execution_integration.json`, identity `71c4171282621555b4567e1204f633a6e58f8e9235a5e99d9d3a993e66a32833`.
- Amendment03 completed the explicit per-arm cold preflight and all22 valid single cold calls. All24 fixed positions remain present, with two proved office1 blocks. Every observed feature/support/label/rank parity passed, maximum feature difference0.0; the full8-scene U2 mean is6.614763293619035 seconds. G1/G3 full-cohort means remain null. Canonical time resumption retained28 exact observation/parent/worker files and did not repeat a physical call.
- Amendment04 consumer changes now pass73 focused tests. They retain172/14 typed positions with actual168/10 scientific coverage, null blocked full-cohort metrics/counts/times and explicit conditional-publication review. Actual diagnostic/table/report generation, measured-PDF QA, the final327-requirement review and normal GitHub publication remain pending. Required172/14/24 totals are unchanged; no global scientific COMPLETE has been claimed.
- Previous scene0056_00 smoke receipts and failed commands are preserved. Explicit pre-freeze regeneration corrected only lossless legacy FP32 storage and environment-dependent probability-audit metadata; no scientific parameter, label decision, extra method or development map changed.

## Amendment06: Full Fixed-Method Analysis And Lossless Publication

Primary ownership: report semantics, packaging design, integration and review.
Allowed source changes: `reports.py`, `publication.py` and the focused contract
tests. The15 scientific producers,168 actual outputs,10 actual ordered pools,
all26 diagnoses, original tables/PDF and22 reserved cold observations stay exact.

- [x] Execute actual26-scene diagnostics and typed tables/reports, retaining
  all172/14/24 positions with168/10/22 complete and explicit technical blocks.
- [x] Complete actual rendered-PDF QA and actual consumer integration audit.
- [x] Diagnose actual213MiB publication failure: repeated full JSON evidence
  does not meet the50MiB target under separate gzip compression.
- [x] Reproduce the omitted Replica U2/G3 Pareto scope with two failing assertions.
- [x] Compare all fixed cohort methods in Pareto analysis; keep the prescribed
  A3 paired deltas and2x2 contrasts unchanged, and disclose unavailable inputs.
- [x] Validate deterministic lossless whole-evidence XZ packaging against every
  original byte/hash; retain direct small tables, summaries and reconstruction
  metadata, with a standard-library extraction interface for archived evidence.
- [x] Record preservation preflight, archive previous reports, run one final
  current-source focused suite and reuse the same real eight-condition smoke;
  freeze the documented consumer amendment before regenerating reports.
- [x] Build and audit the actual complete compact release below50MiB, including
  manifest and eventual primary review. No technical block becomes COMPLETE.
- [ ] Perform the327-requirement primary review, canonical all-resume reuse and
  normal GitHub publication with full local/remote SHA verification.

Amendment06 actual evidence:73 tests passed, the original smoke is exact and
all15 scientific producers are unchanged. The real archive restores1,638 members
and715,826,986 original bytes;1,658 source copies pass byte/hash comparison.
Actual release including its manifest is42,740,397 bytes, before primary review.
Full scientific counts remain blocked at168/172,10/14 and22/24.

Amendment07 fixes the chronological publication audit: only six identified
subsequent push/delivery requirements may be PENDING_PUBLICATION, never a
scientific or implementation gap. Primary read all327 catalogued requirements;
the evidence classification and full code/artifact review remain in progress.

## Amendment09: Canonical Completed-Table Reuse

- [x] Diagnose actual all-resume failure from locked tuple metadata versus
  serialized JSON lists; all original sources and133 worker ledgers are exact.
- [x] Reproduce the production build/reuse failure with a real locked-payload
  regression; the verified canonical-identity comparison passes and still
  rejects changed metadata ordering while preserving existing table bytes.
- [x] Preserve amendment08 release/review/report and failed canonical-command
  artifacts before replacing owned consumer metadata.
- [x] Validate current74 focused tests and unchanged eight-condition smoke;
  freeze the consumer correction at69297ee372fb3ac297a8e221c5a541dbfa87a75a.
- [x] Build the actual amendment09 release at44,250,790 bytes before its new
  primary review. Original scientific, cold and numeric artifacts are exact.
- [x] Review the updated compact release against all327 original requirements:
  292 proven complete,28 scientific-blocked,7 later publication obligations.
  Current release including its primary review occupies45,117,460 bytes.
- [ ] Execute actual canonical all-resume through normal publication and verify
  the full remote SHA. Scientific counts remain168/172,10/14 and22/24.
