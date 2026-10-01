# Recovery Wave 2 Implementation Plan

> Execution stays with the primary agent using superpowers:executing-plans.
> Only explicit, deterministic mechanical leaves may use mechanical_worker;
> scientific decisions, native integration, debugging and final review stay here.

**Goal:** Implement, measure and normally publish the complete specified recovery
study against the measured BB00_NATIVE + D2 baseline, including negative results.

**Architecture:** A new recovery_wave2 package consumes immutable parent receipts
and caches. W changes labels only; U appends outputs on baseline-unowned raw support;
A/S own maps use an isolated native build, staged geometry screening and fresh
semantic anchors. Selection and nomination freeze before new Replica predictions.

**Tech Stack:** Existing Python/NumPy/SciPy/Torch environments, the original
released evaluator, pinned OVI-MAP C++/pybind extension and existing model assets.

**Spec:** CODEX_FINAL_EXECUTION_EN.md, PROTOCOL_SPEC.json,
IMPLEMENTATION_CONTRACTS.md and SOURCE_EVIDENCE.md in this directory.

## Global Constraints

- Base commit: c21297413954ecb7d706d1050a8e07822938ea6f.
- Branch: research/ovimap-recovery-wave2-v1; deployment N0_UNCHANGED.
- Original four development and eight exposed Replica scenes/schedules only.
- Maximum 20 new development + 8 new Replica full map-scene successes.
- No new baseline maps, training, checkpoints, SAM/CropFormer/text forwards or fits.
- Recovery: minimum 100 source rows, maximum 128 candidates, 3 views/candidate,
  32 initially uncached FC image inputs per scene application.
- A: native mode4/data_association2, ratio 0.0, explicit nonmatch USE_NATIVE.
- One GPU model worker under the existing lock; mapping 2 x 8 threads;
  evaluation 3 x 4 BLAS threads, reduced concurrency only for actual resources.
- Parent files are read-only; every miss/receipt goes to the new attempt.
- Geometry and predictions lock before diagnostic GT access; candidate/view
  selection never consumes target coordinates or annotation labels.
- Exact ordered released pooling; APall actual .50-.90 overlap vector;
  official ranks recomputed for every exported owner, including U incumbents.
- Freeze/commit all selected recipes before new Replica predictions/metrics.
- Final four reports, specified compact evidence, normal push and full remote SHA.

## Task 1: Isolated specification and source inventory

Files: this directory; new worktree; binding.py; initial experiment_matrix.json.
Consumes: original ZIP and parent resolved_inputs/maps/readouts/frontend receipts.
Produces: resolved_inputs.json with original identities, resolved read-only paths,
source distributions/order, native binary/patch stack, cache and schedule references.

- [x] Read requirements/contracts/spec and relevant original source boundaries.
- [x] Create named branch/worktree from the exact base; verify supplied checksums.
- [x] Run supplied reference tests as specification checks, not production proof.
- [x] Add read-only path resolver and memoized consumed-object verification.
- [x] Bind actual 12 parent baseline map/readout receipts and assets once.
- [x] Check real consumed paths, complete schedules and original frame policy.

## Task 2: W and append-only U prediction primitives

Files: weight_readout.py, recovery_registry.py, export.py;
tests/evaluation/test_recovery_wave2_light.py.
Consumes: exact parent endpoint decisions, raw/painted support, captured lineage.
Produces: fixed interpolation decisions, capped registry, legal request manifest,
new locked payloads with genuine ranks and preserved incumbent support.

- [x] Write and run failing tests for exact endpoints, all missing-source subsets,
  source-free fallback, physical support ordering and owner-ID renaming.
- [x] Implement W as t=6*gamma-2 interpolation of real endpoints, copying endpoints.
- [x] Implement registry K=(R==owner)&(O==0), no incumbent repair, spatial hash ties.
- [x] Reconcile every segment through aliases against all active final raw owners.
- [x] Test append-only owner/class preservation and complete expanded rank universe.

## Task 3: Genuine recovery sources and capped FC acquisition

Files: recovery_sources.py, recovery_fc_worker.py, binding.py cache inventory.
Consumes: native pickle/request/content receipts, final Q scores/retained requests,
FC dense/region/text receipts and original effective operators.
Produces: U1/UQ/U2/U3 scores, request proofs, deterministic budget and cost receipts.

- [x] Test single-view canonical FP32 classifier against duplicated-view old reader.
- [x] Validate exact saved native observation/request and final reconciled Q lineage
  on actual candidates (10 U1 and 4 UQ restored observations across development).
- [x] Select U2 exact U1 request; U3 top3 lawful captured requests, no substitutions.
- [x] Freeze initial dense inventory and U2-then-U3 union allowance before inference.
- [x] Reuse verified parent text/dense/region content; new pooling/misses stay local.
- [x] Track attempted/used/unavailable/cap exclusions and physical/standalone cost.

## Task 4: Real light evaluation and four-scene cache screen

Files: evaluation.py, workflow.py, run_ovimap_recovery_wave2.py.
Consumes: locked complete predictions; expanded owner metadata with N unavailable.
Produces: official scene rows/pools, actual matches/ranks, recovery funnel/support.

- [x] End-to-end check that every eligible recovered owner enters actual masks,
  manifest and unchanged scorer; do not use a label-only old registry.
- [x] Run all eight fixed development light conditions, reusing true equivalents.
- [x] Record probability/partition/scorer equality separately; aliases need proof.
- [x] Measure paired U1/U2 common success and capped same-covered-subset evidence.

## Task 5: Native per-group actions and A1/A2/A3 maps

Files: association.py, mapping_hooks.py, native_patch.py, native_trace.py;
third_party_patches/ovimap/recovery_wave2_v1/ and isolated upstream/build.
Consumes: own depth-valid preinsert probe and unchanged CropFormer/depth schedules.
Produces: native action integration, trace evidence, twelve development maps.

- [x] Test Hungarian positive-benefit A1 and spatial-tied rowwise A2/A3 planners.
- [x] Implement actions through candidates, fresh labels, mode4 counts, reservations
  and transitive aliases; fallback has no preallocation or blanket prior tokens.
- [x] Build isolated extension on capture + backbone + recovery patch stack.
- [x] Run serial3-valid-frame off/all-fallback/accepted/follower native checks;
  supplement with a real native synthetic fixture if real followers are absent.
- [x] Run all three A arms on four scenes, preserving complete failed attempts.

## Task 6: Cached S1 and complete causal S2 path

Files: sam_completion.py, mapping_hooks.py, after-frame native callback.
Consumes: genuine parent track bits, RAW winner/remap/seed correspondence and
current CropFormer; S2 additionally owns its prior and last two snapshots.
Produces: protected CropFormer completions, causal conflict diagnostics and S maps.

- [x] Round-trip actual raster winner labels to saved tracks; never infer logits.
- [x] Test mutual unique IoU/coverage attachment, standalone overlap and C preservation.
- [x] Construct full development S1 rasters; alias only actual canonical equivalents.
- [x] Implement S2 history capture, same-track visibility warp, stable factor0
  owner evidence, q>.2 addition-only suppression and unknown/history abstention.
- [ ] Run conditional complete S1/S2 four-scene maps where required by input gate.

## Task 7: Geometry screen and own-map semantic regeneration

Files: evaluation.py geometry adapter; workflow.py; task-local source/cache adapters.
Consumes: prediction-only locked raw maps, then separate diagnostic annotations.
Produces: full raw diagnostics and actual standard NATIVE/FC_EQ/D2 readouts unless
the complete four-scene aggregate meets the fixed catastrophe rule.

- [x] Test catastrophe fraction/pp/count units and full-cohort decision scope.
- [ ] Compare all-GT raw recall/IoU/fragments/coverage/F5 with parent definitions.
- [ ] Generate fresh map-specific N/Q/F, own requests/projection and temperatures;
  parent cache is exact lookup only, every new miss remains task-local.
- [ ] Publish semantic NOT_RUN_RESOURCE_SCREEN distinctly from measured geometry.

## Task 8: Single compositions, freeze and eight-scene transfer

Files: selection.py; workflow.py; selection/freezes and measured matrices.
Consumes: complete four-scene ordered pools and standalone logical costs.
Produces: selected gamma/recovery/light package and at most one qualifying map.

- [x] Test feasibility (-.05/-.10/-.10pp), gain (+.20/-.10/-.10pp), bands and ties.
- [x] Measure exactly one selected W x U composition; final package awaits full map screen.
- [ ] If eligible, measure one new-map x light four-scene two-by-two; no A x S sweep.
- [ ] Generate leave-one-development-scene-out fixed-method sensitivity pools.
- [ ] Commit frozen identities/recipes/nomination before every new Replica prediction.
- [ ] Run all eight fixed light Replica conditions and distinct frozen package;
  transfer only the one qualifying map, retaining complete cohort statuses.

## Task 9: Reporting, primary requirement audit and publication

Files: reporting.py; four RECOVERY_WAVE2 reports; compact specified artifact paths.
Consumes: actual rows/pools/funnels/actions/SAM diagnostics/failures/costs/identities.
Produces: code, patch, real compact evidence, final decision and remote proof.

- [ ] Run scoped production tests and actual native/evaluator integration checks.
- [ ] Generate six specified scientific tables and four dynamic reports; report
  actual negative/equivalent/blocked/screened leaves and missing timings honestly.
- [ ] Primary reviews full diff and requirement-by-requirement actual evidence for
  every C0-C11 and execution section; remaining evidence keeps goal active.
- [ ] Verify compact arrays and external restoration manifest, no unrelated changes.
- [ ] Normal push named branch; compare full local HEAD with remote branch SHA;
  write final external publication/final.json without recursive verification commit.

## Validation Commands

Use the existing ovimap-map Python. Production scope:
`PYTHONPATH=src:. python -m pytest -q tests/evaluation/test_recovery_wave2_*.py`.
Actual phase command is the original required command with the verified idle GPU;
report/publish phases never start map/model jobs. The new phase runner is required
to record actual commands, statuses and immutable prerequisite identities.

Current status: 12-scene binding and the complete eight-condition development
light screen are measured. Seventeen scoped production tests pass. Four actual development registries have
14 candidates and 10 with lawful captured views; no target or annotation labels
were opened to create them. Source restoration/budget primitives pass 4 further
scoped tests. Binding found 0 missing artifacts and
indexed 1455 parent dense image contents, 6420 pooled regions and 2 text spaces.
Development recovery performed 10 fresh region poolings, zero image encodings and
zero text forwards. All four official light pools keep baseline APall 12.5817838%
for U; W040/W045 degrade it. Four recovered owners in scene0534_00 enter the actual
official minimum100-target-point manifest. Per-scene AP averages are not used.
The isolated recovery_native_v2 extension passes exact serial3-frame old/new/off/
all-native TSDF, label, owner, alias and raster parity. Forty-one accepted real-frame
pairs and a native two-group/one-owner follower fixture were verified. Its extension
SHA is 48cfc80599c7ec7849749d18eff163680e6c1d937b9e1aa37d73b764f963af1b.
All 800 development S1 frontend frames were checked against original binary tracks
and RAW winner/remap evidence. The changed canonical frame counts are 159, 160,
160 and 159; all positive CropFormer pixels and separate original groups remain.
Five SAM tests, one full-cohort catastrophe test, two native cache/accounting,
two selection and two workflow/freeze-boundary tests pass. Own-map S2
history/probe/warp and addition-only conflict paths are implemented; full S maps
remain pending. All four A1 maps, geometry diagnostics, fresh N/Q/FC readouts and
official pools are measured: D2 APall 11.0744142%, AP50 23.7343537%, AP25 35.5838704%,
mIoU 25.2214135%, mAcc 31.8512985%. A1 passes the geometry screen but is a semantic/
high-IoU-instance tradeoff and does not qualify for new-map transfer.
All four A2 maps and geometry diagnostics are complete. Best-IoU falls 3.08786pp,
but R50 loses only 1 GT; the exact combined catastrophe rule does not trigger.
All A2 and A3 own N/Q/FC readouts and pools are now measured. A2 fixed-D2
APall=11.9978458%, mIoU=24.0460969%; A3 APall=12.1455530%, mIoU=23.7112450%.
Neither meets the prescribed NET_GAIN guard. The prescribed association-screen
CLI has completed with all twelve maps, three full geometry screens and fresh
standard readouts. S1/S2 eight-map execution is running under the same 2x8 CPU
allowance. The prescribed CLI now exists and cache-screen --resume has
actually passed. Standalone additional encoder inputs for all baseline light
arms are zero relative to the actual baseline pipeline; this does not imply
zero standalone baseline cost or zero GPU work for full-map semantic regeneration.
The selected gamma=.5 and NONE produce the one measured RW_LIGHT_COMBO, exactly
equivalent to the measured baseline APall 12.5817838% and other pooled metrics.
Thirty-three targeted production tests pass in current_production.xml. Actual
released matcher diagnostics validate terminal TP/FP-entry parity and retain
numeric tie ambiguity; all twelve A-map planned/realized action traces have been
aggregated. Paired U1/U2 classifications and the real four-scene U2/U3 common-
coverage official pools are measured. Dynamic six-table/four-report generation,
fixed-method leave-one-out pooling and primary-review publication guards are
implemented; final generation awaits complete S screens and Replica transfer.
The pre-commit FC producer bytes were restored with their exact original SHA
7723aef12433579efe4bdc30a8d3b1337bf1ec7301ad1947c3d9aa493080495f and archived
as historical provenance. No original locked FC/readout receipt was overwritten.
The initial diagnostic build, failed follower traces and GPU self-occupancy failure
remain preserved. The remaining S map screen, final compositions, committed selection freeze,
Replica transfer, complete reporting and publication are still pending.
Worktree moved to shared storage after a local-disk
write failure; the original /home/ww/crove/ovimap-recovery-wave2 entry is a symlink.
No improvement or full-study completion claim is made here.
