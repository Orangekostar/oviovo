# T1 cached attribution implementation plan

Goal: execute T1_ATTRIBUTION_SPEC.md on both saved independent FP32 Room0 runs.
Primary agent owns design, scientific semantics, implementation, review and integration.
Use executing-plans inline; the user's autonomous-execution instruction supersedes approval prompts.
Reviewed base: 8cefe6b4bca464e0478f82e6ef1e6d0e3783e347; initial diff empty.
Task worktree: /home/ww/crove/ovimap-t1-attribution.

## Design and fixed boundaries

A single immutable prediction mask pool per run supplies condition records containing only
labels, kept canonical IDs, rank scores and assignment priorities. Native owner zero stays
unknown provenance. Positive unique owner ID is canonical index + 1. No GT arguments enter
prediction generation; all GT correspondences, class filters, traces and regional confusion
belong to the diagnostic runner. Do not load model weights or raw mask logits.

Preserve old fuse_proposals public call and default predictions. Default OVI order is descending
area then owner ID; SF is stable descending released score. Label reuse >= .5 and fusion NMS
>= .7. NMS visit order never changes canonical export order. Source area is point count.
New provenance names OVI with an explicit receipt-bound readout ID rather than hard-coded S1a.

Use the receipt-bound original evaluator with its runtime float thresholds and six-decimal
confidence parser. Instrument only observations in evaluate_matches with AST-inserted callbacks;
unchanged evaluator and traced function must return identical arrays including NaNs. Capture
GT loops, duplicate score events, ignored predictions, FN counts and final PR arrays without
reimplementing its matching policy. Candidate geometry diagnostics remain separately named.

## Tasks and acceptance

- [x] Read spec, required implementation/callers and historical receipts; baseline 8 relevant tests pass.
- [ ] Bind current files from the two FP32/T1/evaluation receipts; hash only consumed assets once.
  Validate exact coordinate/order equality before projection reuse and preserve both run identities.
  Export actual 51/48 vocabularies and evaluator runtime protocol, source hashes and score precision.
- [ ] Add fusion_attribution.py and test_static_t1_attribution.py. Fail first on missing functions;
  fixtures: IoU=.5 reuse / .7 suppression; same-label ownership ties; source-order NMS with
  canonical output; immutable pool; assignment projection commutation; exhaustive region partition.
  Refactor proposal_fusion.py to call the split operations; compare full old arrays/ledger decisions.
- [ ] Add run_static_t1_attribution.py and frozen config. Save one mask bank/IoU cache/run, all
  candidate records including empty/suppressed, source owners, regions and unique maps. Execute
  historical cached smoke before the matrix. Core O/S and U00/U10/U01/U11 first, on both runs.
- [ ] Add evaluator trace module and diagnostic runner. Deterministic fixtures cover duplicate
  events, void/group, region <100, invalid class, no GT and exact runtime threshold comparisons.
  Validate traced/untraced AP and PR/FN observer state; reuse fixed projection and original GT IDs.
  Report overlapping AP, semantic unique map, disjoint unique AP and separate high-IoU diagnostics.
- [ ] Finish fixed geometry bridge with historical emitted registry/scores; report eligibility
  mismatch against core O pool rather than merging it into geometry effects. Add native-vs-S1a
  2x2 control, RAW/CLASS_NORM/OVI_FILL ownership, class-normalized rank only, and SF-first NMS.
  Assignment-only manifest inputs must be identical; rank-only owner/semantic arrays identical.
- [ ] Generate all four machine-readable tables: paired performance; intervention differences and
  interactions; thresholded GT-object gains/losses/borrowing/suppression/evaluator events; regional
  owner/class/correctness transitions and full confusion deltas. Region sums must equal global.
  Deterministic examples include loss categories when present; absent categories stay explicit.
- [ ] Assess repeat agreement by GT-object outcomes, never raw query ID. Decide whether exactly
  one prediction-only repair is justified; otherwise NO_REPAIR_JUSTIFIED. No threshold grid.
- [ ] Final spec sections 1–10 audit, evidence-based decision, measured costs, exact executed
  commands and statuses, one future scene-level confirmation plan. Commit scoped changes/small
  artifacts, push task branch under existing authorization and verify remote SHA.

## Validation and output contract

Use default Python for NumPy-only tests; original released evaluator uses
/home/ww/miniconda3/envs/ovimap-map/bin/python (NumPy compatible with np.in1d).
Relevant baseline tests: test_static_proposal_fusion.py, test_static_projected_masks.py,
test_static_projected_instance_metrics.py, test_static_released_loader.py.
New deterministic tests: tests/evaluation/test_static_t1_attribution.py and evaluator trace tests.
Run a real historical cached parity smoke then both complete matrices. No AP-increase assertions.

Small results: artifacts/static_ovmap/t1_attribution_v1/. Large masks/owners/projections:
/mnt/shared/ww/ovimap-t1-attribution-v1/. No new model inference or mapping, no ScanNet search.
Missing inputs block dependent cells only. No new score model, region repair, GT selection or
cross-scene claims. Final table proportions stay full precision; display percentages and pp deltas.
