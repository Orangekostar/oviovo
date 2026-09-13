# 0913_1 — single-query fallback and full-history recovery attempt

Status: COMPLETE for the Room0 R3 development experiment; no useful AP gain.
Task as a whole remains incomplete (R4–R6 and multi-scene evidence are still pending).

## Result

The independent S2 switch accepts exactly one semantic observation only when its context
crop has depth-valid ratio >=0.9, sharpness proxy >=0.5 and original visible area >=500 pixels,
and native geometry is depth-consistent in at least two independent sampled camera frames.
Each supporting frame needs >=100 rendered owner pixels within strict 5 cm of measured depth.
Camera representatives must differ by >=5 cm translation or >=5 degrees rotation.
These development thresholds were fixed before inspecting the fallback GT result.

| Native ID | Predicted class | Independent support frames | Projected vertices | Released instance evaluation |
|---|---|---:|---:|---|
| 17 | cloth (50) | 6 | 9 | excluded by unchanged minimum region size 100 |
| 34 | ceiling (2) | 11 | 108 | ignored; class outside released 48-class instance subset |
| 103 | bowl (49) | 7 | 173 | ignored; projected region has no valid instance GT |

IDs 68, 110 and 39 have no qualifying independent support and are rejected. The original
71 eligible semantic predictions are preserved; S2 adds three native semantic assignments
and two exported masks. At IoU .25/.50/.75 the added-mask diagnostic has 0 novel TP, 0 counted
FP and 2 ignored masks. Ignored does not mean correct or useful. Full per-object margins,
confidence, thresholds, frame IDs and export outcomes are retained in the receipts.

| Condition | Semantic mIoU | mAcc | Semantic APall | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.332857180 | 0.381586721 | 0.154315629 | 0.343233300 | 0.368233300 |
| S2 | 0.332857373 | 0.381586933 | 0.154315629 | 0.343233300 | 0.368233300 |

The negligible vertex metric difference and unchanged AP are not evidence of successful
recall recovery. Keep S2 disabled in the effective combination by default. No new image
encoder inference or GT-guided prediction was used. Context-bbox quality is disclosed as a
proxy: the original union mask for these retained queries was not present in the old pickle.

## Geometry and compatibility

The producer renders the original native mesh and compares its depth with 200 raw depth
frames. It uses the native integer-pixel convention, not Open3D's default half-pixel offset.
Semantic query counts never supply geometric support. Measurements use final geometry and
are explicitly STATIC_OFFLINE_FINAL_GEOMETRY_VISIBILITY, not causal online evidence.

All 88 mesh-backed native owners, including unknown semantic objects, are retained in the
geometry arrays. C++ color logs omit several historical IDs; consistent cached RGB identity
records are merged without reading their feature values or labels. Omitting these records
caused a one-vertex B0 mismatch (ID 91); the fixed export reproduces original B0 semantic
labels and all released B0 mask/score/class files exactly.

Restoring unknown owners changes the input of class-agnostic evaluation relative to the
legacy semantic-filtered export. Both B0 and S2 use the SAME restored owner array. Their
class-agnostic scores are identical: instance mIoU .5220, mP@.50 .6790, mR@.50 .5978.
Do not count differences from the earlier semantic-filtered B0 (.4589/.7619/.5217) as method
or fallback gains. Geometry coordinates are unchanged; this difference is export/readout coupling.

The final 200-frame support pass took 74.55 s on CPU with eight renderer threads. Raw owner
maps, native vertex IDs and projected owner IDs remain under
/home/ww/oviovo_baseline_runs/20260913_static_ovmap/room0/geometry_support_nativepixels_v2.
They can support the next GT-free multiview graph stage. Preparation cost is separate from
warm readout and historical frontend/VLM cost; no end-to-end FPS claim is made.

## Full-history recovery: locally blocked, other work proceeds

The original temporary pool has 764 cached features; all 592 retained feature arrays match
their temporary files exactly. The remaining 172 lack owner/area metadata. A read-only
instrumented replay with the existing native binding recorded 74 queries through frame 70;
64 overlap retained records and match owner, bbox AND area exactly. It then attempted a bbox
absent from the original query pool: frame 70, (160,492,1199,679). The cache-only guard raised
before any encoder inference. No recovered record has been promoted as full-history truth.

The old Python 3.8 binary is absent at the historical source path. The available CPython 3.11
binding loads, but strict replay/query-schedule parity has not been demonstrated. This is an
observed blocker, not proof of a specific cause such as version drift or nondeterminism.
The wrapper remains available for a compatible binding and preserves the original source
and caches. The first path-layout attempt also failed; its duplicated cropformer directory
was corrected in the wrapper. Neither failed attempt is reported as a completed experiment.

## Verification and next work

87 relevant tests pass, covering the original protocol/readout, native pixel rays, occlusion,
camera independence, single-query gating, no fake duplicate queries, and added-mask diagnosis.
Actual argument vectors and source/input hashes are in artifacts/static_ovmap/room0_fallback.
Prediction entry: run_static_ovmap_readout.py --conditions B0 S2 with --native-binding,
--enrichment and --fallback-support. Semantic/instance evaluators use --projected-owner-array;
GT is opened only by those evaluators and diagnose_static_fallback.py.

Next: R4 merge/split evidence from native owner visibility and original 200 masks; then one
dense branch and separate T3D track. Full-history metadata remains a separate local blocker.
Replica8/ScanNet18 and the final requirements audit are still outstanding. Original source
documents and IMPLEMENTATION_PLAN.md retain the full requested scope.

Validation command (2026-09-13):

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/evaluation/test_static_*.py tests/evaluation/test_ovimap_native.py tests/evaluation/test_run_ovimap_native.py tests/evaluation/test_finalize_ovimap_native.py tests/evaluation/test_run_ovimap_paper_protocol.py tests/evaluation/test_baseline_static_metrics.py
git diff --check
```
