# Local ownership implementation handoff

Six required cells **COMPLETE**; scientific outcome **COMPLETE_NO_NET_UNIQUE_GAIN**. See [four result tables](LOCAL_OWNERSHIP_RESULTS.md), [decision](../../../artifacts/static_ovmap/local_ownership_v1/decision.json), and [full reviewed specification](LOCAL_OWNERSHIP_SPEC.md). Original attribution `NO_REPAIR_JUSTIFIED` is superseded only by this explicitly authorized new method task, not silently edited.

## Source, inputs and execution

Reviewed and actual start HEAD: `98cafc4a96bf906878116ae3284da28bc974ad4d`, clean; branch `research/ovimap-local-ownership-v1`. Isolated worktree `/home/ww/crove/ovimap-local-ownership`. No unrelated branch changes. Execution/final source and configuration SHA256s are in [source identity](../../../artifacts/static_ovmap/local_ownership_v1/source_identity.json); this handoff does not recursively embed its own commit SHA. Final Git push/SHA verification is reported after the commit.

Exact model runs: `fp32_primary` and `fp32_repeat` from the original distinct FP32 T0 caches; use corrected `predictions_verified`, not two rereads counted as extra model runs. [Binding](../../../artifacts/static_ovmap/local_ownership_v1/prediction/input_binding.json) contains consumed-file hashes, T0/query/mask/class/coordinate equivalence and candidate identities. Original native geometry owners come from bound native projection, never G1/G2 owners. Archived graph JSON supplies frame calibration/exporter provenance only. Optional raw logits remain unopened and were not rehashed.

Selected frame IDs: `[0, 60, 130, 190, 260, 320, 390, 450, 510, 580, 640, 710, 770, 830, 900, 960, 1030, 1090, 1160, 1220, 1280, 1350, 1410, 1480, 1540, 1600, 1670, 1730, 1800, 1860, 1930, 1990]`. Selection used only the original200-frame schedule and pose validity/separation, then at most32 evenly spaced temporal indices. All exclusions and original IDs are in binding. Camera1200×680, fx/fy600, cx599.5/cy339.5, depth scale6553.5 are receipt-derived. World points transform via camera-to-world inverse rotation; camera-z and integer-pixel np.rint ties-to-even match existing code. Equal-depth z-buffer ties use original source row; each pixel counts once. Zero masks/invalid depths/occlusion abstain. Predicted masks may share OVI frontend errors.

Environment: `/home/ww/miniconda3/envs/ovimap-map/bin/python`, `3.11.15 | packaged by conda-forge | (main, Jun 11 2026, 03:34:02) [GCC 14.3.0]`, NumPy `1.26.4`; existing native evaluation environment only. No installation or environment merge. Commands below were actually executed from the worktree with exit0; original 34-cell evaluator was not rerun.

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_static_local_ownership.py --config configs/evaluation/ovimap_local_ownership_v1.json --output /mnt/shared/ww/ovimap-local-ownership-v1/predictions
```

prediction: exit0.

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/evaluate_static_local_ownership.py --config configs/evaluation/ovimap_local_ownership_v1.json --predictions /mnt/shared/ww/ovimap-local-ownership-v1/predictions --output /mnt/shared/ww/ovimap-local-ownership-v1/evaluation
```

evaluation: exit0.

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/summarize_static_local_ownership.py --config configs/evaluation/ovimap_local_ownership_v1.json --predictions /mnt/shared/ww/ovimap-local-ownership-v1/predictions --evaluation /mnt/shared/ww/ovimap-local-ownership-v1/evaluation --output /mnt/shared/ww/ovimap-local-ownership-v1/analysis
```

analysis: exit0.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/evaluation/test_static_local_ownership.py tests/evaluation/test_static_t1_attribution.py tests/evaluation/test_static_proposal_fusion.py tests/evaluation/test_static_projected_masks.py tests/evaluation/test_static_projected_instance_metrics.py
```

focused_tests: exit0.

Artifact/report packaging is performed by this checked-in script:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/finalize_static_local_ownership.py --root /mnt/shared/ww/ovimap-local-ownership-v1
```

Its completion receipt is [packaging](../../../artifacts/static_ovmap/local_ownership_v1/packaging_receipt.json). The output directories for the three computational runners must be new; to reproduce, choose a new external root and pass its prediction/evaluation/analysis subdirectories consistently, then package that root. Existing baseline files remain references, not regenerated targets.

## Implementation and protocol

- [Shared evidence](../../../src/static_ovmap/ownership_evidence.py): pose selection, whole-cloud shared visibility, positive-mask sparse histograms and leave-atom-out candidate correspondence. No GT/class input. Usable-view count0 marks missing score; fallback uses0.5 without evidence, while one usable fallback view retains its computed score. Challengers require at least2 views. Scores are smoothed evidence, not correctness probabilities.
- [Local/spatial ownership](../../../src/static_ovmap/local_ownership.py): exact membership bits refine2cm cells; radius1.5cm connected groups, no dense distance allocation. Atom order is centroid x/y/z with deterministic atom-ID ties. Fallback is U00 OVI-first raw source area/canonical tie order. Both local methods consume exactly the same persisted unaries/feasible sets/atoms. Sparse6-neighbor/3cm graph uses at least2 co-visible mask histograms, normalized symmetric weights and bounded sequential descent. No old parent locks, labels/masks/NMS changes or post-splitting.
- [Prediction runner](../../../scripts/evaluation/run_static_local_ownership.py): explicit method registry, unknown methods reject, cache identities bind masks/coordinates/query/run/observations/calibration/atom/evidence/unary/graph/method/source/parameters. [Frozen config](../../../configs/evaluation/ovimap_local_ownership_v1.json) matches the pre-evaluation saved bytes.
- [Evaluator adapter](../../../scripts/evaluation/evaluate_static_local_ownership.py): global FILL checks independent projected global-order parity; local methods check original support/coverage/IDs and fixed projection without fake scalar priorities or global comparisons. Original evaluator/protocol and old T1 runners are unchanged. [Candidate parity](../../../artifacts/static_ovmap/local_ownership_v1/evaluation/fp32_primary/candidate_manifest_parity.json) and [isolation](../../../artifacts/static_ovmap/local_ownership_v1/verification.json) prove new candidate manifests equivalent to U00 and byte-identical across the new family.
- [Post-hoc diagnosis](../../../scripts/evaluation/summarize_static_local_ownership.py): actual released events, U00-added GT retention, OVI losses, nonexclusive error flags, source competing candidate IDs and fixed-graph extent/boundary/component proxies. Fixed U00 region matrices include unmatched pred0 FN; every paired delta sums to global. Predictions are never optimized again on the GT domain.

Post-execution code review found a missing-self duplicate-centroid edge case that could exceed the neighbor cap. A failing fixture demonstrated it; an explicit cap fixes it. Both full real atom sets have zero affected rows, verified in [neighbor-cap evidence](../../../artifacts/static_ovmap/local_ownership_v1/neighbor_cap_verification.json), so graph/evaluator inputs and measured results remain unchanged. Execution-time and final source hashes are both retained; no scientific threshold changed and no benchmark rerun was needed.

## Task-specific completion review

| Requirement | Evidence/status |
| --- | --- |
| §0–3 reviewed base, source/assets, scope | Clean specified base; input binding and actual frame/depth/calibration files verified. Two distinct T0 identities; zero inference/training/mapping. |
| §4 exact six U00 cells | Six COMPLETE new rows; six separately labeled reused baseline rows. No U11 substitution or repeated34-cell study. |
| §5.1–5.2 observations | 32 original-schedule, pose-distinct shared frames; positive-depth/instance z-buffer, integer pixels and source-row tie; real smoke PASS. |
| §5.3 atoms | Full1,862,429-point clouds; exact candidate signatures and bounded connectivity; neutral atoms allow SF within original OVI support. |
| §5.4–5.5 evidence/unary | Leave-atom-out subtraction,16/.20/.05 gating, >=2 challenger views, explicit missing state, beta.05, fallback/canonical ties; sparse arrays and frame correspondences archived externally. |
| §5.6 spatial | Same atoms/unaries/feasible sets, evidence-gated sparse weights, sentinels filtered, lambda.10, <=5 sequential sweeps; actual energy and degree-mass bounds verified. |
| §5.7 parameters | Exact specified config written before new GT evaluation and remains byte-identical; no scientific deviation or post-GT tuning. |
| §6 family/output/evaluator | Local membership/coverage/projection validated without global coercion; global parity preserved; candidate AP identical to U00; no unique rescoring/redistribution. |
| §7 caches/artifacts | Run and query-specific source/evidence/solver hashes, one bank reference per run, source maps and sparse evidence; GT confined to evaluation/diagnosis. |
| §8 focused verification | 29 passing scoped tests plus one real input smoke. Unchanged original trace test campaign not repeated. CPU8/chunk65536, no new packages or large raw logits. |
| §9 tables/interpretation | Four human/machine tables, both gains/losses and same-class transitions, high coverage, energy vs metric distinction, residual FP at99→101, costs and one next experiment. |
| §10–11 GitHub delivery | Both MD files and scoped small evidence prepared with relative links. Mandatory task-branch commit/push and remote SHA comparison are the final operation; final response carries actual full SHA/URL. |
| §12–14 evidence limits/self-review | Geometry-only local adaptation, not a reproduction or independent teacher. All six use U00, same unary keys, no GT-selected behavior. Primary performed review; no delegated scientific decisions. |

## Evidence locations and limits

[Performance](../../../artifacts/static_ovmap/local_ownership_v1/analysis/performance.json), [effects](../../../artifacts/static_ovmap/local_ownership_v1/analysis/effects.json), [objects](../../../artifacts/static_ovmap/local_ownership_v1/analysis/objects.json), [regions](../../../artifacts/static_ovmap/local_ownership_v1/analysis/regions.json), [evidence](../../../artifacts/static_ovmap/local_ownership_v1/analysis/evidence_regions.json), [confusion deltas](../../../artifacts/static_ovmap/local_ownership_v1/analysis/confusion_deltas.npz), [costs](../../../artifacts/static_ovmap/local_ownership_v1/costs.json), [test receipt](../../../artifacts/static_ovmap/local_ownership_v1/test_receipt.json), [event transitions](../../../artifacts/static_ovmap/local_ownership_v1/event_transitions.json), [repeat sensitivity](../../../artifacts/static_ovmap/local_ownership_v1/analysis/repeat_sensitivity.json). New released traces and evaluator ledgers are small compressed files under `evaluation/<run>/<method>/` here. [Archive mapping](../../../artifacts/static_ovmap/local_ownership_v1/archive_sources.json) maps external trace references to their committed copies.

Large owners/atoms/evidence/shared visibility/unique masks stay at `/mnt/shared/ww/ovimap-local-ownership-v1`. [Large-artifact manifest](../../../artifacts/static_ovmap/local_ownership_v1/large_artifacts.json) gives their actual sizes and SHA256s; a reviewer needs that shared filesystem plus original receipt-bound source assets for replay. No weights, original datasets, credentials or raw logits are uploaded. No blocked required cell or remaining implementation task; optional extra methods and future scenes are NOT_RUN by design.

The scientific limitation is a negative result on a single development scene, not missing evidence: high coverage still fails to preserve unique objects. Geometry/class ambiguity, correlated frontend observations, tiny leftover masks crossing evaluator eligibility, same-class extent loss and graph proxy limitations are disclosed. Two saved runs are not significance or scene generalization. Extra3D training is present and scene-exclusion unverified. One next experiment is the separately authorized fixed Room1 confirmation described in [results](LOCAL_OWNERSHIP_RESULTS.md); no launch in this task.
