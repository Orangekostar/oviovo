# Surface Owner Gate Plan

## Goal

Fix large-scale semantic absorption by changing object memory writes from append-only patch accumulation to globally arbitrated surface ownership.

The target failure is:

`bad 2D mask -> lifted wall/floor/ceiling points -> object local_pcd/TSDF write -> dense projection spreads the wrong object label`.

The fix must happen before points enter `local_pcd` or TSDF. Projection-only changes are diagnostic only.

## Core Principle

A scene surface voxel should not be freely owned by multiple objects. Before an object update writes points, every patch point is checked against:

- current TSDF object owner support,
- historical background point cloud,
- current-frame background patches.
- the patch's own strong backgroundness signal, for forced object candidates on large structural surfaces.

Only accepted points are written into TSDF and object `local_pcd`.

## Code Changes

1. Add a surface owner gate inside `src/modules/object_update.py`.
  - Build background voxel support from `state.background.point_cloud` and current-frame `background_patches`.
   - Treat a forced object patch with strong backgroundness as structural evidence unless it is an allowed small attached-surface class.
   - Filter matched patches before `integrate_patch` and `_update_object`.
   - Filter new-object patches before provisional insertion or object creation.
   - Record detailed gate diagnostics in patch metadata and module debug fields.

2. Modify `ObjectUpdateModule.process(...)`.
   - Accept optional `background_patches`.
   - Apply the gate before TSDF integration and before local geometry append.
   - Store rejected structural patches in `last_structural_reject_patches`.

3. Modify `src/pipelines/main_pipeline.py`.
   - Pass `bg_patches` into object update.
   - Feed structural rejected patches back into background update.
   - Add gate stats to frame debug output.

4. Keep dense surface and projection downstream of the filtered observation.
   - `DenseSurfaceModule` should naturally consume filtered observation patches.
   - `run_room0_full_eval.py` projection changes are not the core fix.

5. Add focused tests.
   - Foreign-owner voxels do not enter target object memory.
   - Background-owned voxels do not create new objects.
   - Filtered patches write only accepted points.
   - Dense surface uses filtered observations.

## Configuration

Add under `object_update`:

```yaml
surface_owner_gate:
  enabled: true
  background_enabled: true
  min_accept_points: 20
  min_update_accept_ratio: 0.30
  min_new_object_accept_ratio: 0.55
  max_foreign_owner_ratio: 0.25
  max_background_owner_ratio: 0.35
  attached_surface_classes: [switch, wall-plug, vent]
  attached_max_voxels: 40
  self_background_min_score: 0.65
  self_background_margin: 0.15
```

## Success Criteria

- Object `local_pcd` no longer receives rejected wall/floor/ceiling points.
- TSDF support is integrated only for filtered accepted points.
- Rejected structural points strengthen the background path instead of object memory.
- Room0 audit should show reduced wall/floor/ceiling absorption by object-labeled exports, especially the prior `obj12`, `obj20`, and `obj68` style failures.
