# OVIV2 Replica Static Results (Route 1 Composed)

- Frozen code: `6c9c104c420d6a92759553e5d0507c5855999ef5`
- Base mapping snapshot: Stage 3, 200 frames per scene, source stride 10
- Gate: all six metrics strictly improve for every scene
- Gate result: `8/8` scenes PASS, `48/48` scene-metric deltas positive

## Replica-8 Macro Average

| Metric | Stage 3 | Route 1 composed | Delta |
| --- | ---: | ---: | ---: |
| mIoU | 0.337310 | **0.338303** | **+0.000994** |
| mAcc | 0.411997 | **0.414011** | **+0.002014** |
| f-mIoU | 0.576099 | **0.579900** | **+0.003801** |
| AP25 | 0.249994 | **0.371191** | **+0.121197** |
| AP50 | 0.054591 | **0.124700** | **+0.070110** |
| F@5cm | 0.882351 | **0.885831** | **+0.003480** |

## Replica-7 Held-Out Macro Average

| Metric | Stage 3 | Route 1 composed | Delta |
| --- | ---: | ---: | ---: |
| mIoU | 0.330806 | **0.331811** | **+0.001005** |
| mAcc | 0.408892 | **0.410951** | **+0.002059** |
| f-mIoU | 0.563447 | **0.567597** | **+0.004150** |
| AP25 | 0.245655 | **0.368649** | **+0.122993** |
| AP50 | 0.055279 | **0.123871** | **+0.068592** |
| F@5cm | 0.877530 | **0.881251** | **+0.003721** |

## Per-Scene Results

| Scene | mIoU | mAcc | f-mIoU | AP25 | AP50 | F@5cm | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| room0 | 0.383751 | 0.435431 | 0.666019 | 0.388983 | 0.130507 | 0.917895 | PASS |
| room1 | 0.446825 | 0.555287 | 0.687298 | 0.442444 | 0.088524 | 0.922426 | PASS |
| room2 | 0.353042 | 0.414508 | 0.654910 | 0.359990 | 0.034583 | 0.882230 | PASS |
| office0 | 0.251947 | 0.358613 | 0.425611 | 0.472472 | 0.111310 | 0.861301 | PASS |
| office1 | 0.267219 | 0.359482 | 0.320570 | 0.470211 | 0.387862 | 0.851659 | PASS |
| office2 | 0.347257 | 0.397058 | 0.629922 | 0.456907 | 0.143056 | 0.875404 | PASS |
| office3 | 0.360702 | 0.398825 | 0.636260 | 0.166980 | 0.099976 | 0.885385 | PASS |
| office4 | 0.295685 | 0.392880 | 0.618608 | 0.211538 | 0.001786 | 0.890351 | PASS |

## Verified Replica-8 Comparison

| Method | mIoU | mAcc | f-mIoU | AP25 | AP50 | F@5cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFusion | 0.271 | 0.345 | 0.479 | N/A | N/A | 0.895 |
| ConceptGraphs | 0.118 | 0.212 | 0.106 | **0.401** | **0.238** | **0.897** |
| DualMap | 0.156 | 0.204 | 0.129 | 0.384 | 0.164 | 0.887 |
| **OVIV2 Route 1** | **0.338** | **0.414** | **0.580** | 0.371 | 0.125 | 0.886 |

OVIV2 leads the listed verified baselines on all three semantic metrics. It remains below
ConceptGraphs by `0.0297` AP25, `0.1137` AP50, and `0.0108` F@5cm; these are the remaining
static-map gaps and are not hidden by the semantic gains.

## Frozen Heads

- Instance: connected-component parent/child hypotheses, minimum 20 vertices, child multiplier 8, dedup IoU 0.7, view exponent 2.
- Semantic: entropy power 32 for wall/floor/ceiling/table and 16 for other classes.
- Geometry: extraction weight 0.5; semantic reference weight 1.0; GT-free nearest-reference transfer within 0.075 m.

Verified result JSON:
`docs/paper/results/oviv2/replica/20260721-route1-composed-6c9c104/result.json`

Raw scene artifacts:
`/home/ww/oviovo_experiments/20260721_route1/replica8_frozen_6c9c104`
