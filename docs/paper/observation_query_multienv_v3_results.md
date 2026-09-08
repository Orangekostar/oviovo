# Observation-Query Multi-Environment V3 Results

## Protocol

The comparison uses a frozen Concerto/ReScene backbone with `T=2`, `Q=100`,
seed 45, and the unchanged V2 dense readout. Six disjoint official TRAIN
environments contribute one pair each. Every trained method receives the same
balanced sequence of 2,000 micro-steps, two micro-steps per optimizer update,
and 1,000 optimizer updates. Three exposed DEV environments select one
checkpoint per method; two predeclared CONFIRM environments are opened only
after selection. The primary metric is environment-macro t1 F1@0.50 on
`COMMON_INPUT_SUPPORT_V2`.

## Main comparison

At the common terminal budget, FULL is the only trained method above Frozen.

| Method | Update | Common macro F1 | Micro TP/FP/FN | Micro F1 | Raw mean best IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 0 | 0.06668 | 7/132/61 | 0.06763 | 0.18571 |
| BASE_TUNED | 1000 | 0.05872 | 5/130/63 | 0.04926 | 0.20714 |
| FUSE | 1000 | 0.06460 | 6/128/62 | 0.05941 | 0.19824 |
| FULL | 1000 | **0.07643** | **9/130/59** | **0.08696** | 0.19594 |

DEV selection over updates 0/200/500/1000 chooses BASE at 200 and both
observation methods at 1000.

| Method | Selected update | Common macro F1 | Raw mean best IoU | Full-GT F1 | Identity recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 0 | 0.06668 | 0.18571 | 0.00000 | 0.00000 |
| BASE_TUNED | 200 | 0.06655 | 0.18542 | 0.00000 | 0.00000 |
| FUSE | 1000 | 0.06460 | **0.19824** | 0.00000 | 0.00000 |
| FULL | 1000 | **0.07643** | 0.19594 | 0.00000 | 0.00000 |

FULL exceeds the strongest trained simple baseline, BASE-200, by 0.00987
absolute or 14.83% relative on DEV. It also exceeds Frozen by 14.61%. The gain
comes from two additional true positives, three fewer false positives, and two
fewer false negatives in the micro counts, while FUSE has the highest raw-IoU
diagnostic but a lower final F1. Raw proposal quality therefore does not
determine the final operating point by itself.

## Confirmation

The DEV-selected checkpoints are evaluated unchanged on both CONFIRM
environments.

| Method | Common macro F1 | Micro TP/FP/FN | Micro F1 | Raw mean best IoU |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 0.06140 | 7/163/83 | 0.05385 | 0.14178 |
| BASE_TUNED | 0.06608 | 8/168/82 | 0.06015 | 0.14263 |
| FUSE | 0.04959 | 6/171/84 | 0.04494 | **0.16151** |
| FULL | **0.06957** | 8/160/82 | **0.06202** | 0.15751 |

FULL retains a smaller aggregate gain over BASE: 0.00349 absolute or 5.28%
relative. Both have eight true positives and 82 false negatives; FULL's gain is
entirely eight fewer false positives. The effect is not consistent across
environments.

| CONFIRM pair | Frozen F1 | BASE F1 | FUSE F1 | FULL F1 |
| --- | ---: | ---: | ---: | ---: |
| scene0009_00-scene0009_02 | 0.12281 | 0.11864 | 0.09917 | **0.13913** |
| scene0449_00-scene0449_05 | 0.00000 | **0.01351** | 0.00000 | 0.00000 |

The fixed-view plates preserve both cases in stable pair-ID order. Instance
colors are panel-local, shared GT voxels are neutral gray, and GT never changes
the saved predictions. They are qualitative inspection evidence, not another
metric.

## Mechanism diagnosis

The cached scene0109 diagnosis separates raw, thresholded, exclusive, and
final readout. For Frozen visit 1, 64 non-empty raw queries become two eligible
queries and zero exclusive-query true positives; OVI residual assignment then
produces 2/17/8 and F1 0.13793. Single-environment FULL-1000 similarly falls
from 30 raw queries to three eligible queries and zero exclusive-query true
positives before residual assignment yields 1/18/9 and F1 0.06897. This shows
both weak raw proposals and material score/readout effects.

Same-checkpoint interventions do not identify feedback as the cause. Setting
all feedback betas to zero changes neither update-200 nor update-1000 final F1.
The double attention-and-feedback disable changes update-200 F1 from 0.13793 to
0.14286 and leaves update-1000 unchanged, which is too small and inconsistent
to trigger a separately trained ablation. These rows are inference diagnostics,
not NO_FEEDBACK training results.

## Multi-environment effect and cost

On the shared exposed scene0109 endpoint, the old single-environment
FUSE-1000 and FULL-1000 both scored 0.06897; their multi-environment versions
score 0.14286. This is consistent with reduced single-environment degradation,
but it is only partial causal evidence because the old checkpoints were not
evaluated on the identical three-environment DEV set.

| Method | Train time (min) | Peak GPU (GiB) | Trainable state (MiB) |
| --- | ---: | ---: | ---: |
| BASE_TUNED | 36.41 | 3.72 | 6.95 |
| FUSE | 39.27 | 3.88 | 7.71 |
| FULL | 43.04 | 5.58 | 10.68 |

Across DEV, the mean full-GT surface support is 0.278 and 16 of 164 GT objects
have zero input support. Across CONFIRM it is 0.340 and 9 of 193 objects have
zero support. Full-GT F1 and persistent identity recall remain zero for every
method. The supported-domain gain cannot be described as complete scene
recovery, identity recovery, low Ghost, open-vocabulary generalization, or
SOTA.

## Evidence

Numeric sources are under
`configs/evaluation/results/observation_query_multienv_v3/`. The directory
contains data selection, training curves, checkpoint selection, per-environment
metrics, readout trajectories, interventions, confirmation results, compact
trainable states, and qualitative source manifests. Restricted RGB-D, complete
PLY files, backbone caches, optimizer state, and prediction NPZ files remain
local.
