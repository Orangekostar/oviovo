# OVI-MAP static benchmark implementation plan

Goal: execute 03_CODEX_IMPLEMENTATION_HANDOFF.md without changing evaluation to claim a gain.
Architecture: immutable native geometry and query caches feed an independent offline readout;
GT is loaded only by evaluation. Geometry changes, encoder changes and extra 3D training
are independently selectable tracks. Preserve existing dynamic modules and environments.
Spec: the three source documents in this directory, copied verbatim from docs/0913_1.
Execution: primary agent owns design, core implementation and final review; only explicit
mechanical leaves may use mechanical_worker. User authorized autonomous configuration.

## First delivery: R0–R2

- [x] Fetch verified base d5c0688 and create independent task worktree; preserve dirty source.
- [x] Run existing protocol/export tests: 28 passed.
- [x] Locate Room0 native mesh and 77-instance feature pickle. Stored records are top-10
  sorted by area, NOT temporal order or full query history. Preserve source index for last8.
- [x] Bind available Replica input assets, source changes, vocabulary IDs, model identity and protocol.
  Add protocol_manifest.json and baseline_asset_inventory.json under artifacts/static_ovmap.
  Reuse PAPER_PROTOCOL; add exact ScanNet18 with explicit unknown frame schedules until found.
- [x] Add contracts.py, cache_io.py, observation_bank.py and readout.py under src/static_ovmap.
  Observation identity includes source index, frame and encoder/preprocess identity. Missing
  mask/quality/support stays null. Selection is separate from fusion. Fixed seed 0, K=8.
  Readout consumes one feature space and matching text bank, returns label/score/margin and
  selected indices. last8 retains native order and area normalization sum+1e-6, min queries=2.
- [x] Test native-order weighted fusion against direct original formula; reject equal-dimension
  different spaces; random determinism; quality and directional coverage; min-two behavior.
- [x] Add run_static_ovmap_readout.py. Save B0, random8, S1a/S1b/S1c and all_views separately.
  S1a changes selection only; S1b weights only; S1c both. No extra encoder inference.
- [x] Reproduce original labels using matching text embeddings, evaluate paired native geometry
  and save metrics, changed observations, semantic margins, raw outputs and measured readout costs.
- [ ] BLOCKED: recover historical discarded query ownership/area: all 764 temp feature files exist and the 592
  retained files match exactly, but 172 have no saved owner/area. No extra encoder inference.
  New complete-query capture and its paired Replica8 evaluation are complete, but do not
  recover this historical run; see QUERY_HISTORY_CAPTURE.md and REPLICA8_RESULTS.md.
- [x] Commit and push first real stage; verify remote SHA. R0–R4 stage commits are published;
  the overall task remains active.

## Remaining staged deliverables (must not disappear after S1)

- [x] R3 Room0 fallback: independent geometric support and measured context quality; unknown
  geometry preserved; thresholds and added-instance diagnosis recorded in R3_RESULTS.md.
  Negative result: AP unchanged, S2 off. Additional S2 scenes are NOT_RUN under the prompt
  section F allowance not to run every no-gain branch across all scenes.
- [x] R4 Room0: instance_graph.py, hierarchy.py, native_export.py and split_evidence.py implement
  RGB-D entity evidence, component-wide constrained merge and multiview native surface split.
  Real B0/G1/G2 evaluation and native restoration/export pass; negative/mixed results keep
  G1/G2 off. R4_RESULTS.md records canonical AP separately from released mP/mR and semantic AP.
  Additional G1/G2 scenes are NOT_RUN under section F; the frozen native/readout
  Replica8 matrix is complete and ScanNet inputs remain BLOCKED.
- [x] R5 Room0: released GLA-CLIP + original AnyUp, 172 frames/497 attempted slots,
  ROI and matched-support ROI controls, sparse refinement at T=1/lambda .1/.2/.3,
  exact native B0 parity, full semantic AP, retained storage and measured costs.
  Negative result keeps D1/D2/D3 off; R5_RESULTS.md records support and adapter limits.
- [x] R6 Room0 complete segmentation/checkpoint/SigLIP2-1152, observed RGB-D point cloud,
  FP32 query-chunk runtime, T0/T1 and repeated T0 evaluation. R6_RESULTS.md preserves OOMs,
  extra-3D-training boundaries, score/overlap sensitivity and fixed-seed variability.
  Additional T0/T1 scenes are NOT_RUN; this checkbox covers the Room0 independent
  extra-3D track only, not a Replica8 result.
- [x] Record R7 as optional, NOT_RUN and disabled; no query-conditioned masks were added.
- [x] Keep implemented B0/B1/S1/S2/G1/G2/D1/D2/D3/C1/T0/T1 selectable and status-labelled;
  optional Q0/Q1 are explicitly disabled with null entrypoints, not implemented launchers.
- [x] Replica8 main table and 7-scene diagnostic: 8 native runs, 12 readout conditions,
  released pooled AP and pooled confusion completed; see REPLICA8_RESULTS.md.
- [ ] BLOCKED: ScanNet18 inputs and actual frame schedules. Physical-scene grouping and
  ScanNet200 IDs are locked; unavailable cells remain null+reason, not Replica substitution.
- [x] Complete available main/ablation/cost/per-scene tables with source/config/model/frame identity,
  cache scope, elapsed time, GPU memory and original metric files. Preserve negative results.
  Unmeasured historical VLM/export/peak-memory fields are null with explicit reasons.
- [x] Audit every A–I requirement; verify relevant tests and real scene smoke;
  handoff includes actual commands, stage status, limitations, git commit/push evidence.
  REQUIREMENTS_AUDIT.md retains BLOCKED items; audit completion is not full goal completion.
- [ ] BLOCKED: source package reference/test_static_core.py is absent; cannot run its 8 tests.

## Verified source trace

ovimap_paper_audit.py is called by audit_ovimap_paper_parity.py and finalize_ovimap_native.py.
ovimap_native.bbox_from_instance_map uses inclusive XYXY and is called by audit_native_frame.
ovimap_static_anchor imports temporal lifecycle/snapshot modules and stays unchanged.
run_room0_v2_full_eval.py imports src.v2.pipeline.SemanticMapV2Pipeline, not a paper runner.
semantic_memory.object_identity_semantic_label reads committed anchor identity, not reused here.
External mesh_postprocess_utils uses native stored [-8:] then vis_area/(sum+1e-6), minimum two
queries and canonical cosine text matching. Existing Room0 scores are development-only.
