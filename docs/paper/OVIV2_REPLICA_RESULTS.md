# OVIV2 Replica Results

## Evaluation protocol

OVIV2 is evaluated online on the eight Replica scenes using 200 RGB-D frames per scene at stride 10. Route 1 composes the frozen Stage 3 map with an auditable semantic replay, a protocol-aligned class-agnostic instance head, and GT-free geometry stabilization. Replica-8 is reported as a compatibility split; Replica-7 excludes the development scene `room0`. F@5cm measures geometric reconstruction quality. All numbers below come from the `VERIFIED` frozen Route 1 result.

## Aggregate results

| Split | Scenes | mIoU $\uparrow$ | mAcc $\uparrow$ | f-mIoU $\uparrow$ | AP25 $\uparrow$ | AP50 $\uparrow$ | F@5cm $\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Replica-8 compatibility | 8 | **0.338** | **0.414** | **0.580** | 0.371 | 0.125 | 0.886 |
| Replica-7 held-out | 7 | **0.332** | **0.411** | **0.568** | 0.369 | 0.124 | 0.881 |

## Per-scene results

| Scene | mIoU $\uparrow$ | mAcc $\uparrow$ | f-mIoU $\uparrow$ | AP25 $\uparrow$ | AP50 $\uparrow$ | F@5cm $\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| room0 | 0.384 | 0.435 | 0.666 | 0.389 | 0.131 | 0.918 |
| room1 | 0.447 | 0.555 | 0.687 | 0.442 | 0.089 | 0.922 |
| room2 | 0.353 | 0.415 | 0.655 | 0.360 | 0.035 | 0.882 |
| office0 | 0.252 | 0.359 | 0.426 | 0.472 | 0.111 | 0.861 |
| office1 | 0.267 | 0.359 | 0.321 | 0.470 | 0.388 | 0.852 |
| office2 | 0.347 | 0.397 | 0.630 | 0.457 | 0.143 | 0.875 |
| office3 | 0.361 | 0.399 | 0.636 | 0.167 | 0.100 | 0.885 |
| office4 | 0.296 | 0.393 | 0.619 | 0.212 | 0.002 | 0.890 |

## Comparison with verified baselines

| Method | mIoU $\uparrow$ | mAcc $\uparrow$ | f-mIoU $\uparrow$ | AP25 $\uparrow$ | AP50 $\uparrow$ | F@5cm $\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFusion | 0.271 | 0.345 | 0.479 | -- | -- | 0.895 |
| ConceptGraphs | 0.118 | 0.212 | 0.106 | **0.401** | **0.238** | **0.897** |
| DualMap | 0.156 | 0.204 | 0.129 | 0.384 | 0.164 | 0.887 |
| OVIV2 | **0.338** | **0.414** | **0.580** | 0.371 | 0.125 | 0.886 |

OVIV2 achieves the strongest verified Replica-8 semantic result: compared with OpenFusion, mIoU improves by 0.067, mAcc by 0.069, and f-mIoU by 0.101. The held-out mIoU decreases by 0.006 relative to Replica-8. Geometry remains close to the verified baselines, with F@5cm within 0.011 of ConceptGraphs. Instance quality remains the main limitation: OVIV2 trails ConceptGraphs by 0.030 AP25 and 0.114 AP50.

## Reproducibility

- Verified result: [`results/oviv2/replica/20260721-route1-composed-6c9c104/result.json`](results/oviv2/replica/20260721-route1-composed-6c9c104/result.json)
- Benchmark table: [`benchmark_tables_baselines.md`](benchmark_tables_baselines.md)
- Frozen token registry: [`benchmark_tokens.tsv`](benchmark_tokens.tsv)
- Raw runs: per-scene paths and SHA-256 hashes are recorded in the verified result JSON
- Run ID: `oviv2-replica8-20260721-route1-composed-6c9c104`
- Status: `VERIFIED`
- Protocol deviations: none

## Paper-ready statement

On Replica-8, OVIV2 obtains 0.338 mIoU, 0.414 mAcc, and 0.580 frequency-weighted mIoU, outperforming the strongest verified semantic baseline by 6.7, 6.9, and 10.1 percentage points, respectively. The corresponding Replica-7 held-out mIoU is 0.332, while F@5cm reaches 0.881. Instance reconstruction remains the main limitation, with held-out AP25/AP50 of 0.369/0.124.
