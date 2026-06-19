# SED++ Frontend Bottleneck Analysis

Date: 2026-06-12

## Scope

This report summarizes the last SED++ frontend replacement experiment on Replica `room0`, `200f`, `stride=10`.

Compared runs:

- Baseline: `outputs/tmp_validation/20260612_wwai_online_yoloworld_sam_baseline_s10_200f`
- SED++ cap40: `outputs/tmp_validation/20260612_sedpp_prob_cap40_room0_s10_200f`
- SED++ cap80: `outputs/tmp_validation/20260612_sedpp_prob_cap80_room0_s10_200f`

The SED++ runs disable YOLOWorld anchors and anchor-guided SAM fusion. They use direct semantic components from SED++ as proposals.

## Headline

SED++ successfully removes the old frontend cost, but the current replacement does not speed up the full map at acceptable quality.

- Baseline frontend: `proposal_generation` mean `10.437s/frame`, dominated by `anchor_guided_sam_fusion` mean `9.035s/frame`.
- SED++ frontend: `proposal_generation` mean `0.807-0.897s/frame`; model forward is only `0.115s/frame`.
- cap40 is faster but loses too much coverage: mIoU `0.2213`, f-mIoU `0.4488`.
- cap80 recovers useful weighted accuracy: mIoU `0.3668`, f-mIoU `0.5618`, but wall time is `42:22.95`, slightly slower than the baseline `40:37.90`.

The bottleneck moved from YOLO/SAM frontend fusion to backend proposal consumption: `runtime_vis`, `association`, and `object_update`.

## Accuracy

| Run | mIoU | f-mIoU | mAcc | f-mAcc | Final objects | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOWorld+SAM baseline | `0.5190` | `0.6052` | `0.6231` | `0.6921` | `554` | `40:37.90` |
| SED++ cap40 | `0.2213` | `0.4488` | `0.2697` | `0.5952` | `50` | `26:05.58` |
| SED++ cap80 | `0.3668` | `0.5618` | `0.4202` | `0.6984` | `176` | `42:22.95` |

cap40 is not viable for quality. It is `35.8%` faster than the baseline by wall time, but mIoU drops by `0.2976` and f-mIoU drops by `0.1564`.

cap80 is closer to viable quality. It keeps `92.8%` of baseline f-mIoU, but only `70.7%` of baseline mIoU and is `4.3%` slower by wall time.

Large class-level drops for cap80 are concentrated on instance-like or thin/compact objects:

| Class | Baseline IoU | SED++ cap80 IoU | Delta |
| --- | ---: | ---: | ---: |
| picture | `0.839` | `0.001` | `-0.838` |
| basket | `0.784` | `0.130` | `-0.654` |
| blanket | `0.538` | `0.000` | `-0.538` |
| cushion | `0.439` | `0.000` | `-0.439` |
| book | `0.418` | `0.000` | `-0.418` |
| lamp | `0.748` | `0.336` | `-0.412` |

cap80 gains are mainly on broad semantic classes:

| Class | Baseline IoU | SED++ cap80 IoU | Delta |
| --- | ---: | ---: | ---: |
| floor | `0.371` | `0.731` | `+0.360` |
| door | `0.555` | `0.709` | `+0.154` |
| wall | `0.463` | `0.566` | `+0.103` |

This is consistent with the failure mode: SED++ gives useful dense semantic regions, but direct connected components are not instance-quality masks.

## Timing

`proposal_generation` is a top-level stage and includes its frontend sub-stages. For baseline it includes YOLOWorld, SAM proposals, and anchor-guided fusion. For SED++ it includes SED++ inference and postprocess.

| Stage mean | Baseline | SED++ cap40 | SED++ cap80 |
| --- | ---: | ---: | ---: |
| sec/frame | `12.190` | `7.463` | `12.327` |
| proposal_generation | `10.437` | `0.897` | `0.807` |
| yoloworld_primary | `0.131` | n/a | n/a |
| sam2_proposals | `1.214` | n/a | n/a |
| anchor_guided_sam_fusion | `9.035` | n/a | n/a |
| sedpp_inference | n/a | `0.115` | `0.115` |
| sedpp_postprocess | n/a | `0.782` | `0.692` |
| runtime_vis | `1.905` | `1.225` | `3.700` |
| association | `4.413` | `1.072` | `2.109` |
| object_update | `4.214` | `3.253` | `4.508` |
| dense_surface | `0.246` | `0.309` | `0.377` |

For SED++ cap80, the dominant stages by share of mapping time are:

- `object_update`: `4.508s/frame`, `36.6%`
- `runtime_vis`: `3.700s/frame`, `30.0%`
- `association`: `2.109s/frame`, `17.1%`
- `proposal_generation`: `0.807s/frame`, `6.5%`

For SED++ cap40:

- `object_update`: `3.253s/frame`, `43.6%`
- `runtime_vis`: `1.225s/frame`, `16.4%`
- `association`: `1.072s/frame`, `14.4%`
- `proposal_generation`: `0.897s/frame`, `12.0%`

The SED++ model itself is not the runtime bottleneck. Postprocess is larger than inference because it thresholds and connected-component filters dense semantic maps.

## Proposal Volume

| Metric | Baseline | SED++ cap40 | SED++ cap80 |
| --- | ---: | ---: | ---: |
| raw proposals/frame | `33.80` | `39.73` | `75.84` |
| runtime merged groups/frame | `27.91` | `19.73` | `34.50` |
| patches/frame | `56.97` | `22.45` | `40.55` |
| matched patches/frame | `37.86` | `20.05` | `31.33` |
| new patches/frame | `3.83` | `2.12` | `4.17` |
| final local memory points | `2,235,778` | `1,204,339` | `1,625,646` |

cap40 reduces backend load by starving the map. That is why it is fast and inaccurate.

cap80 sends fewer patches than the baseline, but they are heavier semantic regions and produce expensive visibility/association/update work. The issue is not just proposal count; it is proposal shape and semantic granularity.

## Growth Across the Run

Window means over 20-frame windows:

| Run/window | proposal_generation | runtime_vis | association | object_update | raw props | patches | objects | local memory points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline first20 | `5.780` | `1.404` | `1.196` | `2.312` | `29.35` | `38.75` | `45.40` | `463,177` |
| baseline mid20 | `14.473` | `2.213` | `6.655` | `5.462` | `40.55` | `82.75` | `376.25` | `1,777,470` |
| baseline last20 | `9.812` | `1.824` | `5.174` | `4.422` | `30.40` | `48.60` | `537.75` | `2,168,704` |
| cap80 first20 | `0.820` | `4.067` | `1.670` | `3.689` | `78.85` | `39.70` | `37.95` | `469,024` |
| cap80 mid20 | `0.955` | `4.068` | `2.410` | `5.570` | `80.00` | `51.60` | `102.85` | `1,285,313` |
| cap80 last20 | `0.820` | `3.550` | `2.359` | `3.864` | `73.95` | `34.45` | `166.10` | `1,655,048` |

cap80 starts with high `runtime_vis` immediately because it emits many large semantic proposals before the map has matured. The middle of the run is worst for `object_update`, when matched patches and local memory are both high.

## Correlations

Pearson correlations from the 200 frame metrics:

| Run | Relationship | r |
| --- | --- | ---: |
| cap80 | raw proposal count -> runtime_vis | `0.940` |
| cap80 | matched patch count -> object_update | `0.756` |
| cap80 | local memory points -> association | `0.725` |
| cap80 | total object count -> association | `0.671` |
| baseline | patch count -> association | `0.838` |
| baseline | raw proposal count -> runtime_vis | `0.895` |
| baseline | local memory points -> association | `0.800` |

The backend behavior is consistent across designs: more proposals and more accumulated map state make visibility and association expensive. SED++ cap80 mainly changes where the cost appears by emitting broad semantic regions.

## Bottlenecks

1. Direct SED++ components are not instance proposals.
   They segment semantic regions well enough for large surfaces, but miss or merge compact objects such as picture, cushion, book, basket, and lamp.

2. Global proposal caps are the wrong control.
   cap40 reduces compute by dropping recall. cap80 improves recall but overloads backend stages. The system needs class-aware and geometry-aware budgets, not only `max_proposals`.

3. Structural classes are entering object-centric stages.
   Broad wall/floor/door/window-like regions help f-mIoU, but sending them through object update and association wastes compute and can dominate visibility checks.

4. Postprocess is still CPU-heavy relative to model inference.
   SED++ inference is about `0.115s/frame`; postprocess is `0.69-0.78s/frame`. Connected components over dense maps and loose thresholds are the main frontend-side residual cost.

5. Backend cost scales with proposal shape, not just count.
   cap80 has fewer patches than baseline on average, but larger semantic regions increase `runtime_vis` and `object_update`. The masks need to be split, filtered, or routed before mapping.

6. The current design optimizes the old bottleneck but exposes a new one.
   Removing `anchor_guided_sam_fusion` is successful. The current limiting path is now `runtime_vis + association + object_update`, especially for broad, repeated SED++ proposals.

## Design Implications

The right next design is not pure cap tuning. Use SED++ as a fast semantic prior, then shape what enters the object backend.

Recommended changes:

- Add class-aware proposal quotas: keep structural classes sparse, reserve budget for compact object classes.
- Route large structural masks to structural/background handling, not object update.
- Add per-class thresholds and area ranges. Broad classes need higher area filtering; compact classes need lower confidence but smaller connected components.
- Add geometry filters before backend: area ratio, depth variance, compactness, and max projected size.
- Split large semantic components with depth discontinuities before association.
- Keep a small rescue path for compact uncertain objects only. If SAM is reused, run it on top-K ambiguous components, not as full-frame anchor-guided fusion.
- Re-test a mid-budget config only after class/geometry shaping. A plain cap60 run is unlikely to solve the tradeoff.

## Decision

SED++ is useful as a fast frontend primitive, but the current direct semantic-component frontend should not replace YOLOWorld+SAM as-is.

The next implementation should be a SED++-guided frontend with class-aware routing:

- structural semantic masks go to structural/background paths;
- compact object masks go to object mapping;
- optional SAM refinement is reserved for a small number of ambiguous compact object regions.

Success criteria for the next round:

- `proposal_generation` mean remains below `1.5s/frame`;
- full 200f wall time drops below the baseline `40:37.90`;
- f-mIoU reaches at least `0.58`;
- mIoU recovers toward `0.49+`;
- `runtime_vis + association + object_update` is lower than cap80 while preserving more object classes than cap40.
