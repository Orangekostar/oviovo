# 0913_1 static OVI-MAP development checkpoint

Historical checkpoint at cf4cdf3. See [S1_RESULTS.md](S1_RESULTS.md) and
artifacts/static_ovmap/room0_enriched for the subsequent measured coverage result,
released instance AP, full projection parity and updated remaining work.

This is a partial development checkpoint, not final completion of the prompt.
Worktree: /home/ww/crove/ovimap-static. Branch: research/ovimap-static-benchmark-v1.
Base: d5c0688bc662f8e65455cb9c62909de87c941b43, verified against GitHub.
Existing dirty /home/ww/tmp/oviovo-pr checkout was preserved without edits.

## Actual scope and evidence

Implemented static observation contracts, native cache adapter, five selection strategies,
independent area/quality fusion, exact source-order last8, minimum-two and canonical matching.
Added text cache generation, GT-free prediction runner, and separate semantic evaluator.
36 targeted tests pass, including 28 existing native/protocol regressions.
Actual command: `python -m pytest -q tests/evaluation/test_static_ovmap_readout.py tests/evaluation/test_ovimap_native.py tests/evaluation/test_ovimap_paper_audit.py`.
Full executed text/readout/evaluation argument vectors are saved in JSON `command` fields
under artifacts/static_ovmap/room0_area_only_smoke; invoke text/evaluation with
/home/ww/miniconda3/envs/ovimap-map/bin/python and readout with python from this worktree.

Real large outputs: /home/ww/oviovo_baseline_runs/20260913_static_ovmap/.
The evaluator saves each condition's original/new per-vertex semantic array independently.
Raw observations, selected query IDs, frames, margins, missing fields and timings are saved.
Baseline is bit-exact with existing original projected per-vertex semantic labels;
all 61 instances retained in original semantic-instance evaluation have matching class labels.

| Condition (Room0 development only) | semantic mIoU | mAcc | Scope |
|---|---:|---:|---|
| B0 native last8 | 0.33285718 | 0.38158672 | source-order area weighted |
| RANDOM8 seed 0 | 0.29289871 | 0.34783676 | identical cached-query pool |
| QUALITY8 / S1a / S1b / S1c area degradation | 0.33285718 | 0.38158672 | quality/direction not measured yet |
| ALL_VIEWS | 0.36217547 | 0.41645067 | all retained records, different K budget |

Do not describe the ALL_VIEWS gain as fixed-K selection improvement or Replica8 performance.
Readout CPU warm-cache times are approximately 0.08–0.10 seconds; actual latest values are in
paired_summary.json. Historical frontend/VLM/end-to-end timing is null, not zero.

## Important findings and remaining work

- Cache contains 592 observations for 77 instances (47 with >8), while mapper log reports 764
  executed queries. Export retained top-10 by area and sorted ascending. `last8` is not temporal.
  Full history is unavailable in this pickle. No lost query feature was fabricated/recomputed.
- Quality and camera/object directions are absent in the current adapter. Pose is available,
  but pose alone is not an object-relative direction. Next: compute measurable image/depth quality
  and native-mesh object directions; label cost and missing instance mask evidence explicitly.
- Existing other-task text cache has 41 labels, so it was rejected. Generated native Replica51
  text/canonical embeddings using local original SigLIP, with weight/source/preprocess hashes.
- Historical original model download revision and runtime code diff still need corroboration.
  Native cache currently receives its feature-space binding from the explicit text/source setup;
  strengthen this with a native input identity manifest before general multi-encoder use.
- Area-degraded coverage now falls back exactly to quality ranking if no directions are known;
  this avoids introducing arbitrary tie changes from nonexistent directional evidence.
- Actual voxel size, independent strict-5cm 1NN parity, all 200 frame input assets, original
  inference arithmetic comparison and ScanNet sampling remain to be audited. Reused projected
  native outputs passed coordinate/order and baseline-label checks, not yet independent projection.
- Instance AP and released mP/mR are not yet run. Missing metrics remain null and must be added.
  Primary review must cover native geometry's semantic-filter coupling before R3 fallback.
- The reference/ directory promised by source documents is absent from the supplied package.
  No claim of running its eight tests is made; the new adapter tests are separate.

| Stage | Status | Next evidence required |
|---|---|---|
| R0 | NOT_RUN (partial inventory exists) | full native input/protocol/voxel binding |
| R1 | COMPLETE for Room0 label parity; incomplete arithmetic/projection audit | source formula + native 1NN checks |
| R2 | NOT_RUN (area-only semantic smoke complete) | enriched bank, actual coverage, instance metrics |
| R3 / S2 | NOT_RUN | independent geometry support, TP/FP diagnostics |
| R4 / G1 / G2 | NOT_RUN | RGB-D graph, reversible merge/split evidence |
| R5 / D1 / D2 / D3 | NOT_RUN | one verified released dense branch and ablations |
| R6 / T0 / T1 | NOT_RUN | actual SpaCeFormer segmentation interface and run |
| R7 / Q0 / Q1 | NOT_RUN | optional, off by default |
| B1 / C1 / Replica8 / ScanNet18 | NOT_RUN | version isolation, frozen combination, full scene metrics |

Final completion requires all applicable A–I requirements, not only this smoke. See
IMPLEMENTATION_PLAN.md for the preserved full scope. Push status and commit SHA must be verified
using git ls-remote and recorded externally; this checkpoint does not assert PUSH_VERIFIED.
