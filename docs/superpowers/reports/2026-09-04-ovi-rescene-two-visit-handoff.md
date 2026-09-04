# OVI-MAP x ReScene4D Two-Visit Handoff

Date: 2026-09-04

## Repository And External Identity

| Item | Identity |
| --- | --- |
| Repository | `git@github.com:Orangekostar/oviovo.git` |
| Base branch | `research/crove-benchmark-alignment-audit` |
| Base SHA | `1e849acdcba46ab92f8704a77695ce7caefa1516` |
| Working/remote branch | `research/ovi-rescene-two-visit` |
| Frozen Apartment configuration commit | `96ba693b2996aa19b3e695beeb874f835c4375a9` |
| Final local/remote SHA | commit containing this report; verified after push |
| OVI-MAP | `OVI-MAP/OVI-MAP@f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`, MIT |
| ReScene4D | `GradientSpaces/rescene4d@fb2fe42eb8f1e926567c48eea9acb874e608ee10`, MIT |
| ReScene checkpoint | `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT` |
| Selected neural backbone/grid | Concerto, 0.02 m |

`UPLOAD_STATUS` is finalized only after local `HEAD` equals `git ls-remote`.

## Architecture And Voxel Contract

OVI-MAP is the per-visit mapping backbone and owns dense geometry, instance
segments, and open-vocabulary semantics. ReScene is an optional temporal neural
backbone and may contribute identity/change evidence only. The current composer
uses t1 occupancy first, signed visible-free removal second, and OVI t0 fallback
only for occluded/unobserved space. Semantics and final geometry always remain
under OVI authority.

| Representation | Resolution | Persistent | Trainable/input-only | Final-map authority | Interoperability |
| --- | ---: | --- | --- | --- | --- |
| OVI mapping voxel | 0.01 m | Per visit | No | Yes | Surface samples feed adapter |
| ReScene neural grid | 0.02 m | No | Learned input state | No | CSR reverses every token to OVI samples |
| Legacy CROVE object voxel | 0.05 m | Yes | No | No | Historical diagnostics only |
| Evaluator grid | 0.05 m | No | Metric-only | No | Compares exported current surfaces |

These grids are deliberately independent. Neural tokens are never serialized
as final geometry.

## Protocol And Results

The frozen protocol SHA-256 is
`720952cd75e214de36e318051d6aae18bd762109ba0d3a8ccb73a14f7398e650`.
Pair selection uses only source trajectory, frusta, event metadata, and
visibility coverage, never method scores.

| Scene | t0 | t1 | Status |
| --- | --- | --- | --- |
| Apartment | 766--1021 | 1217--1472 | Executed at configuration commit `96ba693` |
| Office | 2145--2400 | 2446--2701 | `OFFICE_NOT_RUN_HELD_OUT` |

Office could not be executed after the Apartment freeze because the bound
`office/export_manifest.json`, `office/timestamps.csv`, `office/traj.txt`, and
`ground_truth/office` are absent under
`/home/ww/oviovo_benchmark_assets/tesse_cd`. Attempt count remains zero; no
Office outcome influenced selection.

| Variant | Obj. F1 | Dyn. F1 | Chg. F1 | Current mIoU | Ghost | BG F@5cm | Surface F@5cm | Unobserved recall | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | N/A | N/A | N/A | 0.095801 | 0.999846 | 0.000000 | 0.414514 | 0.075068 | PASS |
| B1 | N/A | N/A | N/A | 0.129288 | 0.999846 | 0.157900 | 0.468438 | 0.084523 | PASS |
| B2 | N/A | N/A | N/A | 0.120858 | 0.000000 | 0.445280 | 0.420144 | 0.018912 | PASS |
| B3 | N/A | N/A | N/A | 0.135886 | 0.000000 | 0.366360 | 0.431335 | 0.075646 | PASS |
| B4 | N/A | N/A | N/A | 0.135886 | 0.000000 | 0.366360 | 0.431335 | 0.075646 | PASS |
| B5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT` |
| B6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT` |

| Variant | Final points | Pair/adapter runtime s | ReScene runtime s | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| B0 | 14,621,367 | 0.000 | N/A | N/A |
| B1 | 26,958,216 | 0.000 | N/A | N/A |
| B2 | 12,336,849 | 0.000 | N/A | N/A |
| B3 | 15,622,601 | 0.000 | N/A | N/A |
| B4 | 15,622,601 | 143.514 | N/A | N/A |
| B5 | N/A | N/A | N/A | N/A |
| B6 | N/A | N/A | N/A | N/A |

Object F1, Dynamic F1, Change F1, identity metrics, ReScene runtime, and peak
GPU memory are N/A with explicit reasons in the small result receipt; they are
not zero. B3 is selected over B4 because their outcomes are identical while B4
adds 143.514 seconds. The ReScene verdict is
`RESCENE_BLOCKED_EXTERNAL_ASSET`, not a geometric-baseline substitution.

## Asset And Evidence Status

The current source-bound 3RScan inventory keeps the frozen ten-environment,
44-visit pilot. Twenty-seven visits now contain all four required members; 17
still lack `sequence.zip`. Meshes and annotations are present for those missing
visits, but the RGB-D/pose sequence is required. Status remains
`BLOCKED_DATASET_ACCESS`. The refreshed manifest SHA-256 is
`b3bed41a0d25c7931aaf26ed460778186738f98259281099efcc35b346c7bf1f`.

Large Apartment artifacts occupy 575,211,024 bytes at
`/home/ww/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/apartment-b0-b6-entity-lift-96ba693`.
The matrix summary SHA-256 is
`7aa21465fcf6870a94448b2c34022251bbad79e5dadaed58dbd7ed1d4af26dab`;
it transitively binds all commands, logs, outputs, metrics, and row receipts.

## Required Questions

1. OVI's 1 cm voxel stores persistent TSDF/surface plus label-instance state;
   ReScene's 2 cm grid stores transient learned tokens, so they are not
   interchangeable.
2. The legacy CROVE 5 cm object submap stores one compact weighted local point
   per cell and is retained only as a historical baseline, not final geometry.
3. The evaluator's 5 cm cells are a fixed comparison domain independent of map
   integration and neural sampling.
4. Yes. Deterministic resampling preserves a CSR reverse index to every source
   visit, entity, and point, sufficient for entity-evidence projection.
5. Concerto was selected from the pinned ReScene configuration but was not run.
6. ReScene remains blocked: the pinned official source advertises no checkpoint,
   and no unbound local checkpoint is relabeled as official evidence.
7. The measured t1-only Ghost floor is 0.000000.
8. B3/B4 reach the same 0.000000 floor.
9. B3/B4 gain 0.056735 absolute unobserved recall and 3,285,752 points over B2.
10. Geometric pairing adds no measured value over visibility-only; learned
    ReScene value is unmeasured.
11. Corrected B3/B4 have no remaining Ghost matches. The pre-correction failures
    were exactly 12,010 points from two old Couch residues.
12. Office does not confirm or reject the selection because required assets are
    absent and no attempt was made.
13. 3RScan is still blocked: 17 of 44 selected visits lack RGB-D/pose sequences.
14. Paper-ready claims are the source/voxel contracts and deterministic
    Apartment B0--B4 result. ReScene necessity, Office generalization, 3RScan
    results, tracking/change/identity metrics, and learned performance remain
    diagnostic or blocked.
