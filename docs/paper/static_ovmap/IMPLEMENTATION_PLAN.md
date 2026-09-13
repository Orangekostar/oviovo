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
- [ ] Bind actual input assets, source changes, vocabulary IDs, model identity and protocol.
  Add protocol_manifest.json and baseline_asset_inventory.json under artifacts/static_ovmap.
  Reuse PAPER_PROTOCOL; add exact ScanNet18 with explicit unknown frame schedules until found.
- [ ] Add contracts.py, cache_io.py, observation_bank.py and readout.py under src/static_ovmap.
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
- [ ] Recover discarded query ownership/area: all 764 temp feature files exist and the 592
  retained files match exactly, but 172 have no saved owner/area. No extra encoder inference.
- [ ] Commit and push first real stage; verify remote SHA. Do not call full task complete.

## Remaining staged deliverables (must not disappear after S1)

- [ ] R3 fallback: independent geometric support and observed quality only; preserve unknown
  geometry; explicit threshold config; evaluate added TP/FP and per-object confidence/support.
- [ ] R4 instance_graph.py, hierarchy.py, native_export.py: derive RGB-D whole-object evidence,
  component-wide cannot-link merge then evidence-supported split; reversible mapping and wide IDs.
  Tests: transitive cannot-link, ID export bounds, fixed coordinates, GT-free edge generation.
- [ ] R5 choose one released dense checkpoint after interface/asset checks; separate ROI, pooled
  dense, AnyUp and sparse owner/depth/visibility-gated refinement; T=1 and lambda 0.1/0.2/0.3
  selected on Room0 only; measure K-observation/prototype storage and actual costs.
- [ ] R6 verify complete SpaCeFormer segmentation/checkpoint/SigLIP2-1152 interface, preprocessing
  and NMS; run T0 then T1 with native RGB-D point cloud, or record specific asset/runtime blocker.
- [ ] R7 is optional and NOT_RUN by default; never silently add query-conditioned masks.
- [ ] Keep B0/B1/S1/S2/G1/G2/D1/D2/D3/C1/T0/T1/Q0/Q1 individually selectable and status-labelled.
- [ ] Replica8 main table and 7-scene diagnostic; ScanNet18 uses physical-scene development split,
  explicit sampling, ScanNet200 IDs; unavailable cells null+reason, never substitute Room0 table.
- [ ] Complete main/ablation/cost/per-scene tables with source/config/model/frame identity,
  cache scope, elapsed time, GPU memory and original metric files. Preserve negative results.
- [ ] Final audit against every A–I requirement; verify relevant tests and real scene smoke;
  final handoff includes actual commands, stage status, limitations, git commit/push evidence.

## Verified source trace

ovimap_paper_audit.py is called by audit_ovimap_paper_parity.py and finalize_ovimap_native.py.
ovimap_native.bbox_from_instance_map uses inclusive XYXY and is called by audit_native_frame.
ovimap_static_anchor imports temporal lifecycle/snapshot modules and stays unchanged.
run_room0_v2_full_eval.py imports src.v2.pipeline.SemanticMapV2Pipeline, not a paper runner.
semantic_memory.object_identity_semantic_label reads committed anchor identity, not reused here.
External mesh_postprocess_utils uses native stored [-8:] then vis_area/(sum+1e-6), minimum two
queries and canonical cosine text matching. Existing Room0 scores are development-only.
