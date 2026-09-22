# ScanNet scientific driver implementation plan

> Execute inline with superpowers:executing-plans. AGENTS.md reserves architecture, research semantics, core implementation and review for the primary agent. The frozen specification and current user goal authorize execution; no new design approval is needed.

**Goal:** Finish the native ScanNet S/G/Q study, including actual branch evidence, frozen selection, conditional confirmation and verified publication.

**Architecture:** Preserve the tested mathematical cores. Add explicit dataset/capture adapters and separate branch drivers behind the existing public phase entry point. Prediction construction stays separate from GT target/evaluation work; hashed NPZ/JSON artifacts bind every consumed input. GPU occupancy is a runtime dependency, not a reason to leave CPU implementation unfinished.

**Tech stack:** Existing mapping, CropFormer and alternate-model environments; NumPy/SciPy/PyTorch; pinned native C++ extension and unchanged released evaluator.

**Spec:** `docs/paper/static_ovmap/module_validation_v1/spec/00_CODEX_MASTER_EN.md` through `05_SELECTION_EVALUATION_RELEASE_EN.md`, with constants in `PROTOCOL_SPEC.json`.

## Fixed constraints

- Existing 8 FIT / 2 CAL / 2 SELECT / 2 CONFIRM physical families, 200 scheduled slots each, missing slots never backfilled.
- One visual model on one available GPU at a time, CPU workers <=8; preserve other jobs.
- Native/SigLIP2 FP32, native preprocessing preserved; WOW fixed region-conditioned author interface and allowed BF16/eager settings.
- All S/Q masks/ranks fixed to N0; all G source coordinates/faces/TSDF/projection fixed.
- CAL chooses settings, SELECT chooses modules; CONFIRM remains unopened until a retained candidate and full prediction/code/settings lock exist.
- No global regressions, unrelated baseline reruns, new teachers, parameter sweeps or benchmark changes.

## Task 1 — Complete the native snapshot contract

Files: the scoped upstream C++ header/source and tracked patch; `native_capture.py`, `boundary_jobs.py`, native rebuild CLI, `test_native_capture.py`.

- [x] Add a failing test requiring a true native TSDF snapshot identity, preserving float distance/weight, RGBA, voxel size, block indices and linear voxel order. A surface hash must not impersonate a TSDF hash.
- [x] Add read-only `exportStudyTsdfState()` under the native layer lock, deterministic block order; persist NPZ and hash in capture manifest.
- [x] Verify numerical owner export against the mesh's actual native instance-color lookup, not an invented color-to-ID mapping.
- [x] Rebuild the exact scoped patch in a new native build directory and exercise the changed boundary on the existing two-frame fixture. Preserve native-v7.
- [x] Bind new extension receipt/config; rerun only affected native capture checks.

Evidence: native-v8 receipt at the configured repair root; 9,611 blocks,
4,920,832 TSDF voxels, 1,855,128 vertices with exact numerical-owner/color parity.
Native-v7 versus v8 is not rowwise identical: faces agree, coordinates/owners do
not. Nearest-coordinate audit finds 1,854,809 exact matches, 39 vertices farther
than 1e-4 m, maximum 0.0002550483 m. This is a separate historical replay and no
historical metric parity is claimed. New ScanNet branches bind one shared capture.

Native postprocess finding: 11,696 triangles in this fixture contain mixed
per-vertex numerical owners. Original readout paints every triangle from its
first vertex's color, and drops instances with fewer than two observations.
N0 reconstruction must preserve this original paint; S/Q freeze that exact
owner vector. Numerical voxel owners remain separate capture diagnostics.

Q integration found that v8 exported membership only for currently inserted
segments. ScanNet instance IDs can change non-monotonically, so these sparse
snapshots cannot justify causal owner aliases. Native-v9 adds a read-only scan
of `label_frames_count_` under the same lock, marks `all_known_labels`, and
passes the two-frame native replay (32/32 then 34/31 known/current labels).
All S/G/Q branches will share the v9 replay. First two v8 native captures and
one interrupted third capture are preserved under each scene's
`historical_native_v8`; old study outputs are under `scenes_native_v8`.
They are historical diagnostics, not active result receipts. Raw exports,
annotation/text artifacts and three completed CropFormer outputs are reused.

## Task 2 — Native ScanNet200 readout and evaluation adapter

Files: new `src/static_ovmap/module_validation/scannet_study.py`, `scannet_ground_truth.py`, tests `test_scannet_study.py`.

- [x] Bind 200 class names/IDs from pinned released constants; cache native/SigLIP2 text in their own model spaces with original strings, max length 64 and canonical phrases.
- [x] Read original native feature pickle; reproduce filtering, top10 visibility retention, minimum2 observations, last8 weighted aggregation and canonical classification. Keep every numerical native owner, class0 when unavailable.
- [x] Convert raw annotations through unchanged native conversion/export semantics; freeze strict 0.05m float32 nearest-neighbor projection. Preserve whole GT objects and source row order.
- [x] Construct/lock PredictionPayload before evaluation; freeze N0 serialized ranks, bind true TSDF and source hashes; attach canonical geometry diagnostics separately.
- [x] Test hand-calculated readout/rank fixtures and released trace parity on a real completed capture. Validate changed labels do not change S/Q masks or ranks.

Real scene0547_00 baseline completed: 4,204,503 source rows; exact original
mask/serialized-rank export and released evaluator trace parity. uAP
0.034722222222222224, mIoU 0.13914137886170933. This is one FIT scene,
not SELECT evidence or a module improvement claim.

## Task 3 — S driver

Files: new `semantic_study.py` and leaf CLI; existing `semantic_selector.py` and `region_evidence.py` only for observed integration faults.

- [x] Build <=128 target / <=3 actual captured request manifests, exact masks/bboxes and explicit lineage reconciliation. Preserve excluded/incapable targets with native labels.
- [ ] Run native9/SigLIP2-six/WOW adapters sequentially; save every request result/failure, crop/mask support, raw response/mapping and physical costs. Only encode shared exact identities once.
- [ ] Produce the five direct rows. Freeze the alternative teacher on CAL using uAP/mIoU/cost/ID ordering.
- [x] Separate FIT/CAL target builder: strict geometric IoU>0.5, GAIN/HARM/OTHER and scene weights; support gates exactly as spec.
- [ ] Build32+32 features and all3 fixed heads; preserve FIT-only scalers, checkpoint selection and all6 CAL thresholds; evaluate frozen SELECT rows and correction/damage ledgers.
- [x] Add integration tests for GT isolation, no fallback counted as teacher success and CAL-only selection.

S driver now includes five direct rows, teacher selection, all three fixed
heads, complete six-threshold CAL evaluations, frozen SELECT inference,
separate prediction/event ledgers and checkpoint/optimizer/scaler provenance.
These execution boxes remain open until actual frozen-scene receipts exist.
Twenty affected semantic checks pass, including a five-row driver integration
with the unchanged released evaluator. Native background encoding failure
preserves the successful original six-crop control and marks only context missing.

## Task 4 — G driver

Files: new `geometry_study.py`, reuse `entity_hypotheses.py` / `partition_quality.py`.

- [ ] Build native leaves/surface graph, <=32 captured evidence frames, <=64 disjoint groups and <=8 complete candidates; serialize common pools and provenance.
- [ ] Score every same-pool candidate with agreement; build separate whole-GT local-PQ targets and40-feature head data. Enforce >=20 differing FIT groups / >=4 scenes.
- [ ] Fit fixed head, evaluate all4 CAL margins with canonical AP50/AP75, freeze before SELECT.
- [ ] Export ORIGINAL/AGREEMENT/QUALITY with unique whole-scene ownership; point-count ranks and actual final-mask fresh native semantic rereads, no inherited child labels.
- [x] Test no-op/native equality, invariant TSDF/projection and whole-GT denominator; expose oracle only as a diagnostic.

G implementation now contains the common-pool serializer, strict whole-GT
targets, frozen head/margins, fresh static-mask native six-crop reread and all
three SELECT rows. The v8 first-two-scene pools were generated, then archived
with v8 after the Q membership correction; actual v9 execution remains pending.

## Task 5 — Q driver

Files: new `query_study.py`, reuse `query_state.py` / `query_gain_policy.py`.

- [x] Build candidates from contemporaneous captured rasters/states only; reconcile aliases at their original frame and explicitly block unresolved causal lineage.
- [x] FeatureStore forwards only after budget debit; per-frame barriers; pad missing planned frames with empty candidate lists so quotas retain F=200.
- [ ] FIT seed17/B512 and CAL seed23/B256 exploration; separate visible-GT target raster builder (>=64 valid pixels, >=0.8 allowed class); retain negative NLL gains and enforce support gate.
- [ ] Fit fixed40->32->1 head. CAL freezes strongest B200 nonlearned comparator. SELECT runs four B200 policies; B100/400 curve only if Q_GAIN eligible.
- [x] Test sentinel future/unacquired features and aliases cannot affect prefix decisions; retain physical and logical cost ledgers separately.

Q execution now includes lazy paid FP32 native inference, current-frame GT
target projection outside ranking, FIT/CAL exploration, one frozen utility
head, exact CAL comparator ordering and SELECT inference. Current membership
routes every known paid request by its original complete segment ancestry;
ambiguous split features are discarded without refunds or resurrection.
Missing ancestry-history features receive availability=0. Final registry
reconciliation occurs only after the final barrier. The released-evaluator
integration preserves N0 masks/ranks and exports class0 for unobserved owners.
Eight real v9 frame prefixes reconstruct the exact native candidate universe.
Actual full-scene Q trace/training/SELECT receipts remain pending v9 capture.

## Task 6 — Selection, conditional combination and publication

Files: public study CLI, branch leaf CLI, `selection.py`, `reporting.py`, existing result/handoff documents and scoped artifacts.

- [ ] Replace development smoke placeholders with branch drivers and precise support/resource statuses; hash full transitive artifact inputs for resume.
- [ ] Apply existing frozen selection hierarchy to per-scene rows; support absent learned candidates without treating absence as measured N0 copies.
- [ ] Run only eligible combinations (at most2), requiring actual fresh-mask evidence and the paid hybrid control; block only genuinely unsupported combination prerequisites.
- [ ] Freeze selection/config/code/weights before conditional CONFIRM; at most4 rows per scene, no redesign after outcomes.
- [ ] Generate exactly5 principal result tables, measured costs, bootstrap intervals for positive claims, ledgers/checkpoints and command/source manifests.
- [ ] One final scoped requirement audit; ordinary push and HEAD/remote equality; publication receipt outside the self-referential commit. Complete the goal only when all applicable requirements are evidenced.

Validation commands: `/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest tests/module_validation -q`; scoped `ruff check`; `git diff --check`; leaf real-input receipts followed by public phase receipts. Tests precede implementation for each new meaningful contract; no test-count target.
