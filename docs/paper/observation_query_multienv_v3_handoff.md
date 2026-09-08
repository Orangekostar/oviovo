# Observation-Query Multi-Environment V3 Handoff

## Status

- Branch: `research/ovi-rescene-observation-query-multienv-v3`
- Evidence base: `b39d0182c6e31b22f2110bf5895aa23b223026e1`
- Evaluated head: `241daa6` plus exact source/config/runtime bindings recorded by each run
- Experiment: `MULTIENV_TRAIN_DEV_CONFIRM_COMPLETE`
- Scientific status: `SCIENTIFIC_SIGNAL_POSITIVE_COMMON_SUPPORT_LIMITED`

## Required questions

1. **What did the readout diagnosis find?** Both candidate quality and readout
   matter. On scene0109 visit 1, Frozen drops from 64 non-empty raw queries to
   two eligible queries and zero exclusive-query TP before residual assignment
   recovers two TP. FULL-1000 drops from 30 to three and also has zero
   exclusive-query TP. The evidence is in `diagnostics/readout_stages.csv` and
   `diagnostics/object_trajectories.csv`.

2. **How much official training data was used?** Six distinct official TRAIN
   environments, one T=2 pair per environment, for six pairs total. No DEV or
   CONFIRM environment enters training.

3. **Did multi-pair sampling, caches, accumulation, and resume work?** Yes. Each
   method completed 2,000 micro-steps and 1,000 optimizer updates with gradient
   accumulation two. Exposure counts are balanced at 333 or 334 per pair.
   Pair-specific cache identities, sampler state/cursor, exact resume, and
   checkpoint replay are recorded in the model manifests and
   `training_runs.csv`.

4. **What are terminal and best-DEV results?** At the common 1,000-update
   terminal, macro F1 is 0.06668 Frozen, 0.05872 BASE, 0.06460 FUSE, and 0.07643
   FULL. DEV selection chooses BASE-200 at 0.06655, FUSE-1000 at 0.06460, and
   FULL-1000 at 0.07643.

5. **Did multi-environment training improve over single-environment training?**
   On the shared scene0109 endpoint, FUSE-1000 and FULL-1000 rise from 0.06897
   in V2 to 0.14286 in V3. This cannot isolate data diversity over the full DEV
   set because the V2 checkpoints were not evaluated on the same three
   environments.

6. **Was an extra ablation trained?** No. `I_BETA0` and
   `I_ALPHA0_BETA0` are same-checkpoint inference interventions only. Beta-zero
   did not improve final F1, so no feedback-specific training row is reported.

7. **Does FULL exceed the strongest simple method, and where?** FULL exceeds
   the strongest trained simple baseline, BASE-200, by 14.83% relative on DEV
   and 5.28% on CONFIRM. DEV gains are concentrated in scene0459, where FULL
   has 7 TP versus BASE's 5. CONFIRM scene0009 has 8 TP versus BASE's 7, while
   scene0449 remains a zero-TP failure. Across both CONFIRM environments, FULL
   and BASE have the same TP/FN totals; the aggregate gain is eight fewer FP,
   not additional recovered objects.

8. **Were CONFIRM environments involved in selection?** No. scene0009 and
   scene0449 were declared before V3 model results and were not used for
   architecture, threshold, or checkpoint selection. They are method-unexposed,
   not guaranteed unseen by the pretrained ReScene/Concerto checkpoint.

9. **What are the Full-GT, identity, coverage, and cost limits?** Every method
   has Full-GT F1@0.50 and identity recall equal to zero. Mean input support is
   0.278 on DEV and 0.340 on CONFIRM. FULL training takes 43.04 minutes and
   peaks at 5.58 GiB, versus 36.41 minutes and 3.72 GiB for BASE on the recorded
   hardware. Cached decoder time is not end-to-end OVI mapping time.

10. **Where are the models?** Compact trainable states are tracked under
    `configs/evaluation/results/observation_query_multienv_v3/models/` for
    BASE best/terminal, FUSE terminal/selected, and FULL terminal/selected.
    Frozen ReScene, Concerto, SigLIP, raw RGB-D, full PLY, caches, optimizer
    state, and prediction NPZ files remain local and are referenced by hashes.

11. **What is the strongest paper claim?** With a frozen ReScene/Concerto
    backbone, observation attention plus feedback improves final instance
    discrimination on shared OVI input support over ordinary fine-tuning and
    simple fusion on three DEV environments, with a smaller aggregate gain on
    two predeclared CONFIRM environments. The evidence does not support full
    reconstruction, persistent identity recovery, SOTA, open-vocabulary
    generalization, low Ghost, or multi-seed stability.

12. **What is the single highest-value next action?** Repeat BASE_TUNED and
    FULL with the unchanged six-environment schedule at seed 46, then evaluate
    both on the frozen DEV and CONFIRM sets.

## Reproduction entry points

- Training: `scripts/training/train_ovi_observation_query.py`
- Multi-environment evaluation: `scripts/evaluation/run_observation_query_multienv.py`
- Readout diagnosis: `scripts/evaluation/diagnose_observation_query_v3.py`
- Fixed-view rendering: `scripts/evaluation/render_observation_query_v3_qualitative.py`
- Literal protocol: `configs/observation_query/pilot_multienv_v3.json`
- Frozen split: `configs/observation_query/splits_multienv_v3.json`
