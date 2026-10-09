# Preparation audit notes

These are actual preparation checks and design-review resolutions, not server experiment outcomes.

## Resolved before delivery
- Removed a redundant two-view pose-group average; AREA is now the third meaningful comparator alongside MEAN/SUPPORT.
- Kept the full 26-scene parent distinct from the four-scene SAM-V store; source reader and old mutation-oriented binder have different roles.
- The policy may read fixed predicted classes but never GT classes; corrected the machine-field name to state this distinction explicitly.
- Required pre-acquisition second-view locks, even when shared caches already contain other policies' evidence.
- Separated equal FULL-read budget from DISAGREEMENT's extra anchor subregion/head cost.
- Defined exact surface sites, duplicate/ambiguous faces, deterministic area quadrature, depth/BVH checks and mask-supported versus purely visible sites.
- Kept historical p0 immutable; withdrawal tests concern only actual new records.
- Separated a limited screening-signal gate from the unchanged strict full upgrade gate.
- Defined screen-stop as a completed negative research branch, while real execution blocks remain non-complete.
- Bounded science acquisition, engineering extras and conditional cold timing separately; no per-array-element test-count inflation.

## Actual verification
`python audit/verify_package.py` produced 35 passing specification/document checks. `python -m unittest -v test_reference` in `reference/` ran 12 small NumPy tests successfully. Logs and the machine audit are bundled. Archive content is byte-compared to source files during packaging.

## Not verified here
No actual FC or Open3D integration, access to all shared storage, full-map/scorer experiment, GPU benchmark or GitHub write was performed. The supplied reference kernels are a test aid, not the production implementation. No accuracy/speed gain is claimed.
