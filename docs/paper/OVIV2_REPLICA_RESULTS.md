# OVIV2 Replica Results

## Evaluation protocol

OVIV2 is evaluated online on the eight Replica scenes using 200 RGB-D frames per scene at stride 10. The semantic result uses the frozen `fused_uncertainty` head with an entity weight scale of 0.49. Replica-8 is reported as a compatibility split; Replica-7 excludes the development scene `room0` and is the held-out split used for the generalization claim. Instance AP is class-agnostic, and F@5cm measures geometric reconstruction quality. All numbers below come from the `VERIFIED` Stage 3 result.

## Aggregate results

| Split | Scenes | mIoU $\uparrow$ | mAcc $\uparrow$ | f-mIoU $\uparrow$ | AP25 $\uparrow$ | AP50 $\uparrow$ | F@5cm $\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Replica-8 compatibility | 8 | **0.337** | **0.412** | **0.576** | 0.250 | 0.055 | 0.882 |
| Replica-7 held-out | 7 | **0.331** | **0.409** | **0.563** | 0.246 | 0.055 | 0.878 |

## Per-scene results

| Scene | mIoU $\uparrow$ | mAcc $\uparrow$ | f-mIoU $\uparrow$ | AP25 $\uparrow$ | AP50 $\uparrow$ | F@5cm $\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| room0 | 0.383 | 0.434 | 0.665 | 0.280 | 0.050 | 0.916 |
| room1 | 0.446 | 0.554 | 0.685 | 0.373 | 0.026 | 0.921 |
| room2 | 0.352 | 0.413 | 0.651 | 0.190 | 0.001 | 0.878 |
| office0 | 0.251 | 0.356 | 0.417 | 0.222 | 0.078 | 0.856 |
| office1 | 0.265 | 0.356 | 0.316 | 0.373 | 0.222 | 0.845 |
| office2 | 0.347 | 0.396 | 0.627 | 0.289 | 0.038 | 0.873 |
| office3 | 0.361 | 0.397 | 0.633 | 0.122 | 0.021 | 0.883 |
| office4 | 0.294 | 0.391 | 0.614 | 0.150 | 0.000 | 0.887 |

## Comparison with verified baselines

| Method | mIoU $\uparrow$ | mAcc $\uparrow$ | f-mIoU $\uparrow$ | AP25 $\uparrow$ | AP50 $\uparrow$ | F@5cm $\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFusion | 0.271 | 0.345 | 0.479 | -- | -- | 0.895 |
| ConceptGraphs | 0.118 | 0.212 | 0.106 | **0.401** | **0.238** | **0.897** |
| DualMap | 0.156 | 0.204 | 0.129 | 0.384 | 0.164 | 0.887 |
| OVIV2 | **0.337** | **0.412** | **0.576** | 0.250 | 0.055 | 0.882 |

OVIV2 achieves the strongest verified Replica-8 semantic result: compared with OpenFusion, mIoU improves by 0.066, mAcc by 0.067, and f-mIoU by 0.097. The held-out mIoU decreases by only 0.006 relative to Replica-8, indicating that the semantic improvement is not driven solely by `room0`. Geometry remains close to the verified baselines, with F@5cm within 0.015 of ConceptGraphs. The current limitation is instance quality: OVIV2 trails ConceptGraphs by 0.151 AP25 and 0.183 AP50, so the present result supports a semantic mapping claim but not a state-of-the-art instance reconstruction claim.

## Reproducibility

- Verified result: [`results/oviv2/replica/20260720-stage3-s10-200f/result.json`](results/oviv2/replica/20260720-stage3-s10-200f/result.json)
- Benchmark table: [`benchmark_tables_baselines.md`](benchmark_tables_baselines.md)
- Frozen token registry: [`benchmark_tokens.tsv`](benchmark_tokens.tsv)
- Raw runs: `/home/ww/oviovo_final_outputs/oviv2_replica8_stage3_s049_200f_5f272a8`
- Run ID: `oviv2-replica8-20260720-stage3-s10-200f`
- Status: `VERIFIED`
- Protocol deviations: none

## Paper-ready statement

On Replica-8, OVIV2 obtains 0.337 mIoU, 0.412 mAcc, and 0.576 frequency-weighted mIoU, outperforming the strongest verified semantic baseline by 6.6, 6.7, and 9.7 percentage points, respectively. The corresponding Replica-7 held-out mIoU is 0.331, while F@5cm remains competitive at 0.878. Instance reconstruction remains the main limitation, with held-out AP25/AP50 of 0.246/0.055.

