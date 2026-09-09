# CROVE Fine Current Map V1 Export Path Trace

## Legacy path

```text
run_oviv2_replica.py
  -> derive_labeled_mesh
  -> write_labeled_mesh
  -> evaluation_fused/oviv2_instance_mesh.ply
  -> display-only RGB / instance / semantic recolorings
```

The evidence-bound room0 files under
`$HOME/oviovo_baseline_runs/20260902_crove_dense_recovery/p1/` share exactly
65,350 vertices and 128,412 faces. The RGB file is byte-identical to the fused
stage-3 instance mesh. Therefore, those three files are display readouts of one
coarse geometry, not independent reconstructions. The original user-reported
PLY remains unidentified.

## New static path

```text
native OVI instance_mesh_200.ply (1 cm)
  -> load_native_ovi_surface / verified local cache
  -> bind OVI palette colors to owner IDs
  -> transfer bounded CROVE semantic evidence (S0, S1, S2)
  -> select S2 on DEV mIoU
  -> current_surface_from_visit
  -> one immutable CurrentSurfaceView
  -> select_current_surface once
  -> current_rgb.ply
     current_instance.ply
     current_semantic.ply
     current_state.ply
     current_surface.npz
     current_surface_manifest.json
  -> evaluate the same selected geometry and labels
```

`current_rgb.ply`, `current_instance.ply`, `current_semantic.ply`, and
`current_state.ply` use identical vertex records, face topology and numeric
labels. Only standard display RGB differs. The sidecar preserves the complete
canonical rows, including invalid historical rows for audit.

## New dynamic path

```text
verified OVI Apartment t0/t1 native meshes (1 cm)
  + frozen CROVE B3 coarse provenance (5 cm temporal state)
  -> load/bind per-visit fine surfaces
  -> project current RGB-D evidence to fine rows
  -> map B3 state to source owners
  -> evaluate fine visible-free candidates
  -> resolve current validity without changing the 5 cm state
  -> compose ordered t0+t1 canonical surface
  -> select eligible policy (B3 after candidate rejection)
  -> evaluate and export four views from one CurrentSurfaceView
```

A coarse visible-free cell is only a candidate index. It cannot revoke a fine
historical vertex unless that vertex has valid, visible-free, multi-view depth
evidence. Occlusion, absence from the view frustum and invalid depth cannot
remove it. Current-visit occupied geometry replaces depth-consistent historical
rows.

## Prediction/evaluation boundary

Prediction inputs are OVI native surfaces, RGB-D frames, poses, intrinsic
calibration, CROVE predictions and frozen run provenance. GT meshes, GT labels,
`gt_changes.csv`, and `*_map_gt` artifacts enter only after export for metric
calculation. They do not define vertices, faces, RGB, owner IDs, semantic IDs,
validity or trial selection inputs.

The compact export receipts bind each fixed-view image to the SHA256 and byte
count of its four source PLYs. The large PLYs remain local by policy; the
receipts and figures are committed.
