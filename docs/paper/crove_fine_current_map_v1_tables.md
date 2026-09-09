# CROVE Fine Current Map V1 Table Material

All rows below are development evidence. They must not be copied into aggregate
T1--T4 result cells without matching the table's scene set and protocol.

## Static fine-surface ablation

| Surface/readout | mIoU | mAcc | f-mIoU | AP25 | AP50 | F@5cm | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy coarse CROVE | 0.3828 | 0.4337 | 0.6646 | 0.2804 | 0.0498 | 0.9161 | HISTORICAL_RECORDED |
| OVI 1 cm + S0 | 0.3997 | 0.4566 | 0.6834 | 0.5626 | 0.4707 | 0.9353 | RECOMPUTED |
| OVI 1 cm + S1 | 0.4004 | 0.4558 | 0.6849 | 0.5626 | 0.4707 | 0.9353 | RECOMPUTED |
| OVI 1 cm + S2 | **0.4479** | **0.5111** | 0.6736 | 0.5626 | 0.4707 | 0.9353 | RECOMPUTED / SELECTED |
| OVI source selected at 2 cm | 0.4450 | 0.5088 | 0.6726 | 0.5632 | 0.4465 | 0.9343 | RECOMPUTED_CONTROL |

Caption: Replica room0 development ablation using one native OVI-MAP geometry
and three CROVE semantic readouts. The 2 cm condition is source-surface voxel
selection rather than reconstruction. S2 is selected by mIoU; it does not win
f-mIoU.

## Dynamic current-surface ablation

| Trial | Current mIoU | Ghost | BG F@5cm | Surface P@5cm | Surface R@5cm | Surface F@5cm | Unobserved recall | Observed stale precision | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 t0 native | 0.0958 | 0.9998 | 0.0000 | -- | -- | 0.4145 | 0.0751 | -- | HISTORICAL_RECORDED |
| B1 union | 0.1293 | 0.9998 | 0.1579 | -- | -- | 0.4684 | 0.0845 | -- | HISTORICAL_RECORDED |
| B2 t1 native | 0.1209 | 0.0000 | 0.4453 | -- | -- | 0.4201 | 0.0189 | -- | HISTORICAL_RECORDED |
| B3 legacy current policy | **0.1359** | **0.0000** | 0.3664 | 0.7344 | 0.3053 | 0.4313 | 0.0756 | 0.9310 | RECOMPUTED / SELECTED |
| F1: 2 obs., 2 views | 0.1359 | 0.9226 | 0.3664 | 0.7341 | 0.3056 | 0.4315 | 0.0762 | 0.9209 | REJECTED |
| F2: 3 obs., 2 views | 0.1359 | 0.9258 | 0.3664 | 0.7341 | 0.3056 | 0.4315 | 0.0762 | 0.9201 | REJECTED |
| F3: 4 obs., 3 views | 0.1359 | 0.9379 | 0.3664 | 0.7340 | 0.3056 | 0.4315 | 0.0762 | 0.9176 | REJECTED |
| F4: 6 obs., 3 views | 0.1359 | 0.9531 | 0.3664 | 0.7338 | 0.3056 | 0.4315 | 0.0762 | 0.9350 | REJECTED |

Caption: TESSE-CD Apartment two-visit development ablation. Fine recovery
candidates fail the fixed ghost gate (maximum 0.02), so B3 remains selected.
Object, dynamic-surface and change F1 were not produced by this readout and are
`N/A`, not zero.

B0--B2 are bound to
`configs/evaluation/results/ovi_rescene_two_visit/apartment-96ba693.json`;
B3 and F1--F4 are bound to this task's `metrics_summary.json` and
`trial_log.csv`.

## Cost control

| Readout | Source rows | Offline total time | mIoU delta vs. 1 cm S2 | AP50 delta vs. 1 cm S2 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Native 1 cm S2 | 9,282,303 | full run 361.4 s | 0 | 0 | selected quality readout |
| 2 cm source selection | 370,879 | 27.1 s | -0.0029 | -0.0242 | diagnostic, not reconstruction |

These are offline run-level measurements and must not populate T4 online
seconds/frame or Hz columns.

## Aggregate table audit

The existing T1--T4/S1--S3 templates contain 69 verified values, 306 unfilled
values and 5 `N/A` values:

| Table | VERIFIED | UNFILLED | N/A |
| --- | ---: | ---: | ---: |
| T1 | 33 | 22 | 5 |
| T2 | 0 | 70 | 0 |
| T3 | 0 | 48 | 0 |
| T4 | 36 | 36 | 0 |
| S1 | 0 | 70 | 0 |
| S2 | 0 | 30 | 0 |
| S3 | 0 | 30 | 0 |

This task contributes a room0 T1-style DEV ablation and an Apartment
two-visit T2-style DEV ablation. It does not make the multi-scene T1--T4 tables
complete. Exact value lineage is in `table_lineage.csv`; exact machine-readable
values are in `table_values.json` and `cost_control_2cm.json`.
