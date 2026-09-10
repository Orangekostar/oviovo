# CROVE Entity-Episode Dynamics V2 Results

## Scope and outcome

This report covers one frozen TESSE-CD Apartment development pair, t0 `[766,1021]` to t1 `[1217,1472]`. Every row reads the same 26,958,216-vertex canonical OVI-MAP surface, the same authorized t1 RGB-D window, and the same 5 cm evaluator. Ground truth is evaluator-only.

The measured outcome is `STATE_CORRECTNESS_REPAIRED_NO_MAP_GAIN`. The entity-episode update restores a small amount of valid history without introducing Ghost, deletion error, or confirmed-free recovery, but it does not improve current mIoU over frozen B3. The global selection rule therefore retains `H1_B3 / D1_B3`.

## Development results

| Configuration | Current mIoU | Ghost | BG F@5cm | Surface P/R/F@5cm | Correct new coverage | Accepted relations | Eligible |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| H0 T1-only | 0.1209 | 0.0000 | **0.4453** | **0.7843** / 0.2869 / 0.4201 | 0.0000 | 0 | no |
| **H1 B3 (selected)** | **0.1359** | **0.0000** | 0.3664 | 0.7344 / 0.3053 / 0.4313 | 0.0000 | 0 | **yes** |
| H2 inherit | 0.1357 | 0.0000 | 0.3664 | 0.7347 / **0.3097** / **0.4357** | **0.0295** | 0 | yes |
| H3 +G1 | 0.1357 | 0.0000 | 0.3664 | 0.7347 / **0.3097** / **0.4357** | **0.0295** | 14 | yes |
| H4 ReScene ID-only | **0.1359** | **0.0000** | 0.3664 | 0.7344 / 0.3053 / 0.4313 | 0.0000 | 10 | yes |
| H5 +ReScene epoch | 0.1357 | 0.0000 | 0.3664 | 0.7347 / **0.3097** / **0.4357** | **0.0295** | 10 | yes |
| H6 memory-last | 0.1357 | 0.0000 | 0.3664 | 0.7347 / **0.3097** / **0.4357** | **0.0295** | 12 | yes |
| H7 memory-bank | 0.1357 | 0.0000 | 0.3664 | 0.7347 / **0.3097** / **0.4357** | **0.0295** | 12 | yes |

The H2 update restores 2,939 t0 source rows. Of 40,960 current-supported samples not already covered by t1 prediction, 1,207 are recovered (`2.9468%`). Relative to B3, surface recall increases by `0.004347` and surface F@5cm by `0.004374`; current mIoU decreases by `0.000152`. Deleted-supported rate and bad-recovery rate remain zero. This is a useful state-correction signal, but not a headline map-quality gain.

G1 accepts 14 of 88 candidates, ReScene accepts 10 of 94, and memory accepts 12 of 12. H3, H5, and H6 produce the same scored state as H2, so neither geometric nor learned identity evidence adds a map increment under the row-local action gate. H4 is exactly equal to B3 for current mask, per-point semantics, and headline metrics, satisfying the identity-only invariance contract.

## Model and confirmation evidence

One exact Apartment pair-bound ReScene forward was reused: pair SHA-256 `b75b7a6cb03083e7ab8f3a39d075b52c1162d13d433ac86668c615c6facae4fa`, checkpoint SHA-256 `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`, recorded runtime `3.8139 s`, and peak memory `10,968,595,456 bytes`. Training count is zero. The memory experiment is truthful offline replay on final OVI support; the real asset contains only one final observation, so the bank row is not a real multiview comparison.

Office confirmation was attempted after DEV selection and without threshold retuning. Its status is `RAW_MISSING`: the frozen Office ROS2 database, `gt_changes.csv`, and RGB-D export manifest are absent. No Office metric is reported.

These values are visit-final snapshot diagnostics, not official sequence Obj/Dyn/Chg scores. They cover one DEV pair, have no multi-seed uncertainty, and do not support a SOTA or cross-dataset claim. The figure and exact source rows are under `configs/evaluation/results/crove_entity_epoch_v2/`.
