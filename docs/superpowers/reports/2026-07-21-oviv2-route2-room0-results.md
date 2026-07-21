# OVIV2 Route 2 Replica room0 Results

**Date:** 2026-07-21  
**Scene:** Replica room0, 200 materialized frames, source stride 10  
**Decision:** no Route 2 Pareto winner; keep the Stage 3 baseline for the paper-facing path

## Decision

The hybrid frontend improved room0 instance AP substantially without changing geometry,
but its promoted `sam_labeled` run regressed all three semantic aggregates at 200
frames. The strict Route 2 gate therefore failed. The immutable Stage 3 snapshot remains
the accepted input to Route 1. The failed hybrid run is retained as diagnostic evidence
and may supply ideas for a later instance-only head, but it is not a headline result.

## Frozen Inputs

| Input | SHA-256 |
| --- | --- |
| Replica benchmark manifest | `8547aa8b7dbf7ef7a97f4b516f6557a1d163107404fbfc7ccbd56747c5dc95a4` |
| Materialized frame manifest | `e8efe415039ddea3659fac06fa5b94a459944fba8bddbc91088ffcbc733e79cb` |
| YOLO frontend manifest | `f883cf38c154ff6d2ef18f96bfa4b4aee74af8c9f86f7e2928f8182981d81797` |
| RADSeg dense manifest | `cab154605984b81bbfbd505056dd19c28601f879b9637b3f8dce167c28905fcb` |
| Canonical SAM gate | `3cc109d7f529011f400a3f2ac3baf8eec088ff6d479f63b46b28f65c363adabb` |
| Canonical SAM generator | `7faf64705a80f920eb269b3c842319fc9eccb1e32eda1ed1b23dbe40ce064110` |
| SAM runtime patch | `74bd99e9c0a097be802b3a30f013a8fcbbc74863bae37cb78d1e3f3cb5e9dd87` |
| ViT-H-14 CLIP checkpoint | `9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4` |

Frozen config and cache manifest hashes:

| Variant | Config SHA-256 | Algorithm hash | Cache manifest SHA-256 |
| --- | --- | --- | --- |
| Stage 3 baseline | `f35610ae818038a4e43a754302076b2142e6b18d74e587ad5c337dc50c33f5dd` | `3b756469bca8d2efc9fbb6317f318e080740ffa2b7e935f2215b742d1db9feae` | `f883cf38c154ff6d2ef18f96bfa4b4aee74af8c9f86f7e2928f8182981d81797` |
| `sam_labeled` | `444aeaada267a8d261949c7980dc7d2d81c5f0b03bc53c4b12a2f7545f494c16` | `533d00b108bda288172f891b192bc37c67a14696b3954bea959db01cc4007be2` | `78a5b992c8cafb96f90f2abe48442c3605f3f551887f661515417c4a80821158` |
| `yolo_novel_sam` | `9f673bcc512d6482f3422d0abb73b62bd4bd178b09d68fcddb0cdb6e118f1068` | `babba4c3d6c387109748391332974e24d7cc5ba94c76f5a1983e32dd5b7bdad6` | `0acd2bb3bce89a48588c3e244817833b75482c2f2f4f611f7a11d39dd48da8c1` |
| `quota_nms_ensemble` | `9af1f1b398cceb4cb292421f0ce8825f3e512728c610fc5efceabdbd667389a3` | `a84b3ee16a9c1ee6c9206fef2c795e7c8c10cccb191b81b9e39fd9ad4a58bd56` | `05d7e46cfdaf8c9ed486ba0eb1204417ec20d9ec804c666848613d1af1ecf7c8` |

The 20-frame runs recorded repository commit `91e44e1368c72887bb629c4050c6a34da6d79162`.
The 200-frame run recorded `20c1a1d53ac438497bd7bb1ced09394a7bca42ce`.
The intervening commits changed paper-planning documents only; mapping code and frozen
algorithm hashes were unchanged. All runs recorded dirty-state digest
`65f90056a9066a0d26a5df4d31c06abc17793f3ce0e1d809064c32d278d667f1`
from pre-existing untracked paper directories.

## Cache Audit

All three caches contain 200 frames. Every frame was reloaded against its manifest
checksum. There were no empty frames, no emitted `wall`/`floor`/`ceiling` labels, and
the maximum feature-norm error was below `3.0e-8`. The original YOLO cache has 4,580
proposals and a median of 23 proposals per frame.

| Variant | Min | Median | P95 | Max | Total | Size | Build time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sam_labeled` | 10 | 31.0 | 45.05 | 53 | 6,371 | 32 MiB | 388.01 s |
| `yolo_novel_sam` | 22 | 44.0 | 60.10 | 64 | 8,772 | 43 MiB | 462.90 s |
| `quota_nms_ensemble` | 22 | 43.5 | 59.0 | 64 | 8,635 | 43 MiB | 423.02 s |

| Variant | YOLO | Inherited SAM | RADSeg SAM | Compact | Unlabeled reject | Structure reject | Duplicate reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sam_labeled` | 0 | 4,835 | 1,536 | 0 | 196 | 2,128 | 85 |
| `yolo_novel_sam` | 4,580 | 2,683 | 1,509 | 0 | 196 | 2,128 | 2,104 |
| `quota_nms_ensemble` | 4,578 | 2,558 | 1,499 | 0 | 197 | 2,129 | 1,899 |

The quota cache respected 64 proposals per frame and 12 proposals per class. Its
additional rejection totals were 485 class-cap and 17 global-cap rejections. The
unlabeled fractions were 2.98%, 4.47%, and 4.63%, respectively. The first-20-frame
lifting drop fractions were 0.43%, 0.28%, and 0.28%, all below the 25% rejection gate.

## 20-Frame Gate

| Variant | mIoU | mAcc | f-mIoU | AP25 | AP50 | F5 | Entities | Overmerge | Fragment | s/frame | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh baseline | 0.147151 | 0.166415 | 0.272264 | 0.038854 | 0.000000 | 0.505488 | 32 | 9 | 7 | 8.494 | reference |
| `sam_labeled` | 0.151370 | 0.168691 | 0.272621 | 0.117090 | 0.040441 | 0.505488 | 47 | 8 | 12 | 8.916 | PASS |
| `yolo_novel_sam` | 0.148322 | 0.168630 | 0.272219 | 0.082126 | 0.044118 | 0.505488 | 64 | 13 | 14 | 9.983 | FAIL: f-mIoU -0.000045 |
| `quota_nms_ensemble` | 0.148358 | 0.168677 | 0.272239 | 0.091776 | 0.044118 | 0.505488 | 65 | 13 | 14 | 9.387 | FAIL: f-mIoU -0.000025 |

All runs were finite, completed without exceptions, produced non-empty observation
streams, stayed below three times the baseline entity count, and preserved F5 exactly.
Only `sam_labeled` satisfied the non-regression promotion rule.

## 200-Frame Gate

| Metric | Stage 3 baseline | `sam_labeled` | Delta | Check |
| --- | ---: | ---: | ---: | --- |
| mIoU | 0.3828341530 | 0.3810158903 | -0.0018182627 | FAIL |
| mAcc | 0.4337307984 | 0.4320380603 | -0.0016927381 | FAIL |
| f-mIoU | 0.6646659615 | 0.6642319990 | -0.0004339626 | FAIL |
| AP25 | 0.2803600107 | 0.3274767331 | +0.0471167224 | PASS |
| AP50 | 0.0497723659 | 0.1506332888 | +0.1008609228 | PASS |
| F5 | 0.9160999937 | 0.9160999937 | +0.0000000000 | PASS: IEEE-identical |
| Geometry precision | 0.9708186687 | 0.9708186687 | +0.0000000000 | identical |
| Geometry recall | 0.8672204691 | 0.8672204691 | +0.0000000000 | identical |

The baseline has 92 entities, 31 overmerged entities, 18 fragmented ground-truth
instances, and 3.717 mapping seconds per frame. `sam_labeled` has 136 entities, 27
overmerged entities, 27 fragmented instances, and 4.938 seconds per frame. The run
completed in 17:50 wall time with 1.57 GiB peak RSS and no swap.

The machine-readable gate is
`/home/ww/oviovo_experiments/20260721_route2_frontend/runs200/sam_labeled/pareto_audit.json`
and has status `FAIL`. No failed variant was promoted. Winner reproduction is therefore
not applicable.

## Commands

Cache construction used `scripts/build_oviv2_hybrid_frontend_cache.py` with each frozen
config, matching `--variant`, and its fixed cache output. Functional and full runs used:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/run_oviv2_replica.py \
  --config configs/oviv2_replica_room0_route2_sam_labeled.json \
  --output /home/ww/oviovo_experiments/20260721_route2_frontend/runs200/sam_labeled

/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/compare_oviv2_pareto.py \
  --baseline /home/ww/oviovo_final_outputs/oviv2_replica8_stage3_s049_200f_5f272a8/room0/evaluation_fused/metrics.json \
  --candidate /home/ww/oviovo_experiments/20260721_route2_frontend/runs200/sam_labeled/evaluation_fused/metrics.json \
  --mode route2 \
  --output /home/ww/oviovo_experiments/20260721_route2_frontend/runs200/sam_labeled/pareto_audit.json
```

## Handoff

Route 1 starts from the unchanged Stage 3 baseline. The highest-value evidence from
Route 2 is that finer SAM masks can raise AP25/AP50 by about 0.047/0.101 while preserving
geometry, but direct use of the same observations in the semantic owner path introduces
small semantic regressions. Route 1 should therefore separate the instance proposal head
from the semantic map and tune semantic fusion independently under the six-metric Pareto
gate.
