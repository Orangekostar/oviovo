# Structure Dense Overlay Fast Experiment

## Run

- Experiment: `20260530_room0_structure_dense_overlay_stride10_200f`
- Config: `configs/room0_fast_structural_overlay_4090.yaml`
- Frames: `200`
- Stride: `10`
- Output: `outputs/tmp_validation/20260530_room0_structure_dense_overlay_stride10_200f`
- Evaluation source: `room0_dense_geometry_instance_projected.ply` with conservative structural overlay fusion.

The run completed all 200 frames and wrote the expected exports:

- `room0_instance_map.ply`
- `room0_instance_map_dense_surface.ply`
- `room0_instance_map_tsdf_backbone.ply`
- `room0_dense_geometry_fused_rgb.ply`
- `room0_dense_geometry_instance_projected.ply`
- `room0_structural_overlay.ply`

The tmux wrapper had stale `finished_at` and empty `exit_status` values because shell variables were expanded too early. Completion was verified from `frame_metrics.jsonl`, exports, and `room0/run_report.json`.

## Metrics

| experiment | mIoU | stage total | wall | window | blinds | ceiling | floor | door | chair | table | sofa | rug |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structure overlay 200f | 0.194 | 742.4s | 0.417 | 0.002 | 0.491 | 0.647 | 0.321 | 0.515 | 0.269 | 0.148 | 0.330 | 0.152 |
| fast hybrid 200f | 0.279 | 723.0s | 0.345 | 0.000 | 0.384 | 0.450 | 0.388 | 0.271 | 0.810 | 0.702 | 0.493 | 0.487 |
| async delayguard 200f | 0.274 | 1070.7s | 0.347 | 0.000 | 0.384 | 0.456 | 0.386 | 0.282 | 0.807 | 0.594 | 0.481 | 0.451 |
| 0526 anchorcover rerun2 | 0.549 | n/a | 0.595 | 0.249 | 0.770 | 0.697 | 0.637 | 0.843 | 0.811 | 0.709 | 0.645 | 0.611 |
| 0526 weak structure | 0.546 | n/a | 0.574 | 0.249 | 0.677 | 0.709 | 0.609 | 0.842 | 0.816 | 0.714 | 0.649 | 0.580 |

## Overlay Diagnostics

- Structural overlay voxels: `55063`
- Structure anchors: `1172`
- SAM proposals consumed by overlay: `13476`
- Accepted structure anchor/SAM pairs: `5171`
- Voted pixels: `17327103`
- Overlay candidate dense points: `175060`
- Overlay replaced dense points: `112741`
- Overlay protected dense points: `62319`

## Timing

Total stage time was `742.4s`, close to the fast-hybrid stage total (`723.0s`) and faster than async delayguard (`1070.7s`).

Largest stages:

- `association`: `364.7s`
- `depth_refinement`: `84.6s`
- `proposal_generation`: `58.5s`
- `runtime_vis`: `58.2s`
- `object_update`: `58.0s`
- `yoloe_supplemental`: `47.9s`
- `structural_overlay`: `29.5s`
- `active_set`: `24.0s`

The overlay itself is not the dominant runtime cost. Association remains the main bottleneck.

## Conclusion

This iteration partially validates the idea but should not be accepted as the best version.

What improved:

- `wall` improved from fast hybrid `0.345` to `0.417`.
- `blinds` improved from `0.384` to `0.491`.
- `ceiling` and `door` also improved.
- `window` became non-zero, but only `0.002`, far below the 0526 runs.

What regressed:

- `chair` dropped from `0.810` to `0.269`.
- `table` dropped from `0.702` to `0.148`.
- `sofa` dropped from `0.493` to `0.330`.
- `rug` dropped from `0.487` to `0.152`.
- Overall mIoU dropped from fast hybrid `0.279` to `0.194`.

The root issue is that conservative fusion is still too broad in dense space: it protects points already projected to known objects, but it replaces many points that the object projection leaves unlabeled or structure-labeled. Those points still include object surfaces, so the overlay improves stuff classes by consuming object evidence.

## Next Step

Keep the structural overlay, but make fusion object-aware instead of purely dense-label-aware:

- Treat overlay as a candidate semantic layer, not an overwrite layer.
- For each dense point, block structural overwrite when nearby object support is strong, even if that exact dense point is currently unlabeled.
- Add distance-to-object or visibility-owner protection around `chair`, `table`, `sofa`, and `rug`.
- Use structure overlay mainly on geometry that is large, planar, and repeatedly structure-labeled across frames.
- Give `window` a more targeted rule, because the current broad structure overlay helps wall/blinds but still does not isolate window.

This suggests the next iteration should be a protected-object-neighborhood structural overlay, not a more aggressive overlay.
