# OVI-ReScene Observation-Query Results

## Scope

This bounded pilot keeps the OVI dense D geometry, native ReScene M domain, 100 queries, frozen Concerto backbone, T=2 input, fixed query threshold 0.3, and existing dense/identity protocols. The only real metric run is the frozen native baseline on the previously inspected DEV pair `scene0109_00-scene0109_01`.

The official TRAIN RGB-D `sequence.zip` files are absent. Therefore `OBS_BASE_TUNED`, `OBS_LATE`, `OBS_FUSE`, `OBS_ATTN`, `OBS_FULL`, and `OBS_PER_VISIT` are `NOT_RUN_MISSING_TRAIN_ASSETS`; their table cells are unavailable rather than zero.

## Frozen Baseline

| Visit | Domain | TP/FP/FN @0.50 | F1@0.50 | Raw best IoU | Raw AR50 / AR25 |
|---:|---|---:|---:|---:|---:|
| 0 | Full GT | 2 / 29 / 12 | 0.0889 | - | - |
| 0 | Common M | 2 / 24 / 10 | 0.1053 | - | - |
| 0 | Camera visible | 6 / 24 / 8 | 0.2727 | - | - |
| 0 | Raw query | - | - | 0.1778 | 0.0714 / 0.3571 |
| 1 | Full GT | 2 / 20 / 8 | 0.1250 | - | - |
| 1 | Common M | 2 / 17 / 8 | 0.1379 | - | - |
| 1 | Camera visible | 6 / 15 / 3 | 0.4000 | - | - |
| 1 | Raw query | - | - | 0.1349 | 0.1000 / 0.3000 |

Full/common endpoint identity has 13 persistent GT objects, zero predicted correct links, undefined precision, and recall 0. The forward took 5.236 s and peaked at 1,100,249,088 allocated GPU bytes. Results are in `configs/evaluation/results/observation_query/runs/20260907_scene0109_obs_base_frozen_v2/results.csv`; logits and dense owner arrays are in the adjacent `predictions.npz`.

The current runner reproduced its own masks/classes exactly across repeated runs. It was not bit-exact to the older historical cache: maximum absolute differences were 34.9940 for mask logits and 3.25388 for class logits. This is recorded as `NOT_EXACT_CURRENT_RERUN`; no unsupported root cause is claimed.

## Mechanism Evidence

The current-source T04 rerun confirms that `observations=None` is exactly the native path under an identical RNG state. FULL consumes 490 regions and 51,825 CSR edges, supports 30,131/40,076 M rows, applies 12 observation-attention layers and 13 feedback points, and returns finite mask/class/region logits. Cached decoder runtime is 0.168 s and excludes backbone computation.

The current-source T05 rerun matches all 9 DEV criterion targets. At update 200, losses are CE 0.33305, mask 0.21234, Dice 0.45084, region 4.64836, consistency 5.62168, and weighted total 38.34394. This is criterion validation, not training or performance improvement.

## Current Map

T09 reassigns all 820,617/788,946 source D rows into exclusive new instances or background, without changing XYZ. Shared query IDs are emitted only as `uncertain` candidates. The real baseline has no candidate relation after conflict filtering and identity recall is 0, so the existing t1-first composer interface is implemented but map effect is `NOT_EVALUABLE`. No Ghost reduction or improved current-map quality is claimed.

## Conclusion Boundary

This pilot establishes implementation correctness, real observation support, dense readout, and a negative frozen-baseline diagnosis. It does not establish that the observation-query method improves instances, identities, Ghost, open-vocabulary generalization, or runtime. The next decisive experiment is one legally materialized official TRAIN pair followed by the fixed 200-update smoke and the predefined `OBS_BASE_TUNED`/`OBS_FUSE`/`OBS_FULL` comparison.
