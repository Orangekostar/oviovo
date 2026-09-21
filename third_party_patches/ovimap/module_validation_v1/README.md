# OVI-MAP module-validation capture patch

This patch targets exactly OVI-MAP commit
`f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`. It adds two read-only,
lock-protected numerical export methods and three capture boundaries on the
active mapping path. The native view-selection function remains unchanged;
all-candidate enumeration is a separate pure function.

Check or apply it to a clean checkout:

```bash
third_party_patches/ovimap/module_validation_v1/apply_patch.sh --check /path/to/OVI-MAP
third_party_patches/ovimap/module_validation_v1/apply_patch.sh /path/to/OVI-MAP
```

Capture is disabled unless both variables are set:

```bash
export OVIMAP_MODULE_VALIDATION_CAPTURE_ROOT=/absolute/output/root
export OVIMAP_MODULE_VALIDATION_HELPER_ROOT=/home/ww/crove/ovimap-module-validation
```

The output contains immutable frame PNG/NPZ/JSON records and a final
`surface.npz`. Surface owners use the same
`getInstanceLabel(segment_label, 0.1f)` mapping as the native instance mesh;
the exporter traverses the generated instance-mesh blocks and repeats the
native block/voxel lookup used to color those exact vertices. IDs are never
recovered from mesh colors.

The isolated build and real two-frame parity receipt is written to
`/mnt/shared/ww/ovimap-module-validation-v1/tooling/native_patch_receipt.json`.
It records the compiler, source, patch, loaded extension, capture, and mesh
identities together with the numerical parity counts.
