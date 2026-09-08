# Observation-Query Multi-Environment V3 Plan

## Question

Test whether region observations improve two-visit instance formation beyond frozen ReScene, ordinary decoder fine-tuning, and simple feature fusion.

## Fixed protocol

- Backbone: frozen Concerto/ReScene, T=2, Q=100.
- Training: six disjoint official TRAIN environments, one pair per environment, seed 45, balanced sampling, 1,000 optimizer updates and identical exposure order.
- Methods: `OBS_BASE_TUNED`, `OBS_FUSE`, and `OBS_FULL`, all initialized from the same base checkpoint.
- DEV: scene0109, scene0359, and scene0459. Select one checkpoint per method by macro `COMMON_INPUT_SUPPORT_V2` F1@0.50, then raw IoU, then earlier update.
- CONFIRM: predeclared scene0449 and scene0009, unseen by V3 architecture, threshold, and checkpoint selection. Evaluate only Frozen and the DEV-selected checkpoints.
- Report both `COMMON_INPUT_SUPPORT_V2` and `FULL_GT_V2`; never reinterpret common-support gains as full-scene recovery.

## Execution gates

1. Diagnose raw-to-final readout loss and run same-checkpoint feedback interventions.
2. Prepare six source-bound training pairs and verify alternating-pair/resume behavior.
3. Train the three methods for the same budget and evaluate checkpoints 0/200/500/1000 on the same DEV environments.
4. Freeze checkpoint selection before opening CONFIRM results.
5. Run independent OVI-MAP reconstruction, observation preparation, and evaluation for both CONFIRM environments.
6. Package code, compact results, trainable weights, and provenance; verify the GitHub branch SHA.
