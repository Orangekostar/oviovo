# OVI-MAP x ReScene B7 Results

Date: 2026-09-04
Evaluated commit: `be02e08ec5517cb6c7126a042410da56a2258692`
Decision: `STOP_B7_MAPPING_EXTENSION`

## Apartment B7-G

The single pre-registered full Apartment run completed successfully. B7-G
accepted 5 of 18 attempted registrations and replaced 1,792 unwarped B3 points
with 1,792 visibility-gated registered points. None covered a previously
uncovered GT surface voxel. The measured vector is exactly equal to B3.

| Metric | B3 | B7-G | Delta |
| --- | ---: | ---: | ---: |
| Current mIoU | 0.1358861593 | 0.1358861593 | 0 |
| Ghost | 0.0000000000 | 0.0000000000 | 0 |
| BG F@5 cm | 0.3663600098 | 0.3663600098 | 0 |
| Surface precision | 0.7344127782 | 0.7344127782 | 0 |
| Surface recall | 0.3053313226 | 0.3053313226 | 0 |
| Surface F@5 cm | 0.4313354117 | 0.4313354117 | 0 |
| Unobserved recall | 0.0756464685 | 0.0756464685 | 0 |
| Observed stale precision | 0.9309634046 | 0.9309634046 | 0 |
| Final points | 15,622,601 | 15,622,601 | 0 |

Safety held: 326 visible-free candidates and 283,818 t1-occupied candidates
were rejected; introduced confirmed-free matches and revealed-background
conflicts were both zero. The extension failed the pre-registered Surface-F1
and unobserved-recall gain predicates.

## Conditional Stops

- ReScene identity: `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`; no official final
  checkpoint was found in the clean pinned upstream checkout.
- Learned B7: `STOPPED_BY_B7_G_NO_GO`; no GPU experiment was run.
- 3RScan: `BLOCKED_27_OF_44`; no final identity ranking is claimed.
- Office: `HELD_OUT_ZERO_ATTEMPTS`; no Office result was inspected.

## Evidence

The immutable result package is
`/home/ww/oviovo_baseline_runs/20260904_ovi_rescene_b7/apartment-be02e08`
(36 MB). Its run-manifest SHA-256 is
`7a731691f01b8c548ac144e57097ca4afa376c8c182f871cb3413fd370ff8c0a`.
The compact receipt is
`configs/evaluation/results/ovi_rescene_b7/apartment_b7_g_v1.json`.

After evaluation, the arbitrary-point adapter was isolated from the frozen
B0--B6 `two_visit_execution.py`; that file is again byte-identical to the base
SHA. The evaluated result remains bound to commit `be02e08` and config SHA
`09a62abf...`. The post-evaluation compatibility config is explicitly marked
as not used to generate metrics.

## Verification

- B7 plus legacy focused regression: 110 passed.
- Full repository suite: 5,991 passed, 10 skipped, 0 failed; five warnings are
  existing NumPy 2.5 test deprecations.
- Current B7 source/frozen binding validation: passed.
- Frozen B0--B6 visibility source SHA: `956be641...`, byte-identical to base.
