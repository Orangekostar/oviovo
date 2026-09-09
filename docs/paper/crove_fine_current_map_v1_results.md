# CROVE Fine Current Map V1 Results

## Scope and protocol

This report covers the frozen development experiments for the fine current-map
readout. It does not replace the aggregate T1--T4 benchmark tables. The runs
used a worktree based on `4870b928960783e8d66e9e5e6b2c751d27889400`; the
evaluated implementation snapshot was then integrated as `635e13b`. No model
was trained.

The prediction surface comes from native OVI-MAP meshes. CROVE contributes
current validity, ownership and semantic readout. Ground-truth geometry and
`*_map_gt` data are used only by the evaluator. The temporal state remains at
5 cm; the published surface remains at the native OVI-MAP resolution.

The user-visible problematic PLY was not identified. Consequently, the legacy
65,350-vertex Replica room0 export is reported only as
`USER_PLY_NOT_IDENTIFIED/EVIDENCE_BOUND_REPRESENTATIVE`.

## Static DEV: Replica room0

The native 1 cm OVI-MAP surface contains 9,282,303 vertices and 3,094,101
triangles. All four current-map views use the same selected geometry. S2 is the
selected semantic strategy because it maximizes development mIoU.

| Readout | mIoU | mAcc | f-mIoU | AP25 | AP50 | F@5cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy CROVE coarse export | 0.3828 | 0.4337 | 0.6646 | 0.2804 | 0.0498 | 0.9161 |
| Native 1 cm + S0 nearest | 0.3997 | 0.4566 | 0.6834 | 0.5626 | 0.4707 | 0.9353 |
| Native 1 cm + S1 surface vote | 0.4004 | 0.4558 | 0.6849 | 0.5626 | 0.4707 | 0.9353 |
| Native 1 cm + S2 owner fallback (selected) | **0.4479** | **0.5111** | 0.6736 | 0.5626 | 0.4707 | 0.9353 |
| Native source selected at 2 cm | 0.4450 | 0.5088 | 0.6726 | 0.5632 | 0.4465 | 0.9343 |

Relative to S0, S2 improves mIoU by 0.0482 and mAcc by 0.0545, while f-mIoU
drops by 0.0098. RGB is valid for 97.42% of vertices. S2 changes 992,940 labels
(10.70%) from S0 and leaves 0.34% unknown. The 2 cm row is deterministic
source-surface voxel selection, not a new reconstruction; it keeps 370,879 of
9,282,303 source rows and is therefore only a cost/quality control.

The legacy RGB, instance and semantic files are byte-distinct recolorings of
the same 65,350 vertices and 128,412 faces. They contain 82 entity IDs and 31
semantic IDs. This supports the diagnosis that the old coarse appearance is
primarily a representation/readout limitation, but it does not identify the
user's original PLY or prove that every semantic boundary is clean.

## Dynamic DEV: TESSE-CD Apartment

The two-visit canonical surface contains 26,958,216 vertices. The selected B3
readout exports 15,622,601 current vertices and 5,197,828 triangles.

| Trial | Min. absent obs. | Min. views | Current mIoU | Ghost | BG F@5cm | Surface P/R/F@5cm | Observed stale precision | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| B3 legacy | 1 | 1 | 0.1359 | **0.0000** | 0.3664 | 0.7344 / 0.3053 / 0.4313 | 0.9310 | **yes** |
| F1 | 2 | 2 | 0.1359 | 0.9226 | 0.3664 | 0.7341 / 0.3056 / 0.4315 | 0.9209 | no |
| F2 | 3 | 2 | 0.1359 | 0.9258 | 0.3664 | 0.7341 / 0.3056 / 0.4315 | 0.9201 | no |
| F3 | 4 | 3 | 0.1359 | 0.9379 | 0.3664 | 0.7340 / 0.3056 / 0.4315 | 0.9176 | no |
| F4 | 6 | 3 | 0.1359 | 0.9531 | 0.3664 | 0.7338 / 0.3056 / 0.4315 | 0.9350 | no |

The fine recovery candidates add only 604--13,504 historical t0 rows and fail
the frozen maximum-ghost gate of 0.02. Only 2,341 of 2,960,199 projected
visible-free candidate rows have valid RGB evidence (0.079%). This explains why
the attempted recovery is not trustworthy. The selected dynamic policy is
therefore B3, not a fine candidate. Object, dynamic-surface and change F1 are
`N/A` in this development readout because the required object/change evaluator
contract was not executed here.

## Evidence and limitations

- Compact evidence: `configs/evaluation/results/crove_fine_current_map_v1/`.
- Large static maps: `$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1/current_map/`.
- Large dynamic maps: `$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/apartment_two_visit_full_v1/current_map/`.
- Fixed views: `figures/static_dev_four_views.png`,
  `figures/dynamic_dev_four_views.png`, and
  `figures/legacy_static_display_control.png` inside the compact evidence root.
- D4 confirmation is blocked: the configured OVI run has no matched static
  confirmation surface and no Office t0/t1 native pair. 3RScan meshes are not a
  substitute because no matched CROVE reference/evaluator is bound.
- Run-level elapsed times are not comparable with T4 online per-frame timing.
- `MODEL_UPLOAD_STATUS=NOT_APPLICABLE_NO_NEW_TRAINING`.

The highest-value next experiment is to obtain or build the frozen Office
native t0/t1 OVI pair and run the selected B3-backed export there. Further
Apartment threshold search is not justified by the current evidence.
