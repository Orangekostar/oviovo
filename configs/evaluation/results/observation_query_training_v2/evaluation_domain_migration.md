# Observation-query evaluation-domain migration

The V2 evaluation keeps the 5 cm voxelization and IoU thresholds at 0.50 and
0.25, but corrects the target domains used by the earlier pilot.

- `FULL_GT_V2` evaluates every declared ground-truth target without clipping it
  to reconstructed or model-input support.
- `COMMON_INPUT_SUPPORT_V2` clips ground truth and predictions to one fixed
  method-independent support set derived before inference. Coverage counts and
  support fractions are reported with every result.
- `CAMERA_VISIBLE_INTERSECT_D_V1` is retained as a diagnostic name because the
  current cache samples visibility through reconstructed D; it is not presented
  as full native-surface camera visibility.
- `RAW_FULL_GT_V2` and `RAW_COMMON_INPUT_SUPPORT_V2` report raw-query best IoU
  and recall before dense instance readout.

The former `FULL_GT_LEGACY` implementation clipped targets to D and is therefore
archived conceptually as `RECONSTRUCTION_SUPPORTED_GT_V1`. Its values, including
raw-query values, are not used as a V2 improvement baseline.

For every final-instance domain, V2 computes
`F1 = 2 TP / (2 TP + FP + FN)`. A positive denominator with zero true positives
produces numeric `0.0`; null is reserved for metrics that are not applicable or
were not executed.

The corrected zero-update baseline is
`20260908_scene0109_obs_base_frozen_v2_domains`, evaluated on the exposed DEV
pair `scene0109_00-scene0109_01`. Its source rows are preserved under
`baseline_current_v2/` and summarized in `baseline_metrics_v2.csv`.
