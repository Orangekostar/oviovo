# CROVE Dense Moved-Anchor Readout Design

## Decision

Implement Option A from the approved recovery design: retain each moved bound
OVI-MAP entity as a dense immutable template and translate it so its anchor
centroid equals CROVE's latest causal current centroid. This phase changes only
the emitted current geometry. It does not change temporal state, identity,
lifecycle, dynamic evidence, geometry epochs, association, or the 5 cm state
voxel size.

## Alternatives Considered

1. **Centroid translation (selected).** Apply
   `p_current = p_anchor + (c_current - c_anchor)` to every anchor point. This
   matches the current translation-dominant motion model, preserves the dense
   shape exactly, and requires no unverified correspondence or rotation.
2. **Temporal SE(3).** Rejected for P4A because the frozen anchor has no proven
   object-local frame correspondence with the temporal epoch pose. Applying a
   rotation would introduce an unverified coordinate-frame assumption.
3. **Dense observed reservoir.** Deferred to P4B because it changes runtime
   state, memory, update policy, and capacity behavior. It is unnecessary for
   testing the measured moved-anchor density discontinuity.

## Readout Contract

`compose_anchor_checkpoint()` receives an explicit moved-geometry mode:

- `temporal_compact` is the default and preserves the frozen behavior.
- `anchor_centroid_translation` is the P4A candidate.

For a bound entity already accepted as moved by the existing overlay state:

1. Reuse the immutable anchor point array as the geometry template.
2. Use the latest current-frame `TemporalExportSample.centroid_xyz` when one is
   available in the checkpoint interval.
3. Otherwise use the causal temporal prediction's point centroid. This fallback
   never consults a future frame, ground truth, or evaluator output.
4. Translate the anchor template only. Copy semantics, lifecycle, first/last
   seen, temporal identity, and dynamic-state provenance from the existing
   temporal prediction.

The candidate adds these metadata fields:

```text
geometry_authority = ovimap_anchor_template
geometry_source = causal_ovimap_anchor
state_authority = crove_temporal
template_anchor_id = <anchor entity ID>
transform_source = current_export_centroid_translation
                     or temporal_geometry_centroid_translation
readout_resolution_m = null
readout_resolution_source = native_ovimap_mesh_not_declared
```

`readout_resolution_m` is deliberately null: the native OVI-MAP mesh does not
declare a metric sampling resolution, and the 5 cm temporal/background voxel
size must not be mislabeled as its resolution.

## Runner and Artifact Separation

The composition runner exposes two orthogonal, validated fields:

```text
moved_geometry_mode:
    temporal_compact | anchor_centroid_translation

readout_role:
    formal_baseline | visualization_shadow | evaluation_candidate
```

Allowed pairs are:

- `temporal_compact + formal_baseline`
- `anchor_centroid_translation + visualization_shadow`
- `anchor_centroid_translation + evaluation_candidate`

The default call remains the frozen compact baseline. The selected pair is
recorded in the published run manifest. Shadow and evaluation outputs use
different output roots; neither overwrites the existing composed run.

## Failure Behavior

- Unknown mode/role or an invalid pair fails before output publication.
- Missing/non-finite anchor or temporal centroids fail closed.
- A moved entity without either a current export sample or nonempty temporal
  geometry fails closed rather than silently using an arbitrary transform.
- Existing atomic staging, input revalidation, and no-overwrite behavior remain
  unchanged.

## Tests and Gates

Tests must prove:

- default compact positions and metadata remain unchanged;
- dense output positions equal the anchor points plus the exact centroid delta;
- dense output preserves temporal semantic and lifecycle fields;
- unchanged, occluded, removed, and new entities are unaffected;
- all required geometry/state provenance fields are present;
- mode/role validation fails before creating output;
- the manifest binds the selected mode and role;
- T1 exactness and non-interference remain green.

The real Apartment shadow audit reports, per moved entity, anchor/compact/dense
point counts, dense-to-compact nearest-neighbor median and p90, bounding boxes,
5 cm coverage, and serialized sizes. Promotion to an evaluation candidate is
allowed only after the shadow keeps state/identity metadata unchanged and
removes the measured dense-to-sparse point-count collapse.
