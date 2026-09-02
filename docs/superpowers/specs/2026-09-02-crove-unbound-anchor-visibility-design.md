# CROVE Unbound-Anchor Visibility Design

## Goal

Reduce the measured Apartment Ghost error caused by persistent, unbound
OVI-MAP object anchors without weakening the frozen static anchor, changing T1,
or using ground truth at runtime.

## Measured Scope

P2 attributes 1,592,890 of 2,152,474 official Apartment Ghost matches
(74.002752%) to unbound OVI-MAP anchors. The selected intervention therefore
applies only to anchors that were not bound to a CROVE temporal identity at the
causal initialization cutoff. Bound anchors, new CROVE entities, background,
association, semantics, and the immutable anchor artifact are unchanged.

The one Apartment development pilot uses only RGB-D frames, camera poses, and
timestamps at or before the current frame. Ground truth and evaluator outputs
are offline selection evidence only and never enter the runtime path.

## Visibility Contract

Each unbound anchor is voxelized at 5 cm. Sorted unique voxel keys are retained
in full up to 1,000 keys; larger anchors are sampled at deterministic evenly
spaced indices. Every post-cutoff frame projects those keys into the current
depth image with a 0.10 m depth tolerance.

A frame is strong visible-absence evidence only when:

- at least 10 projected voxels have valid depth;
- at least 80% of valid projected voxels lie in observed free space; and
- no projected voxel agrees with the anchor surface.

A frame is strong presence evidence only under the symmetric 80% rule with no
free-space disagreement. Mixed, occluded, out-of-view, and invalid-depth frames
are neutral and cannot suppress an anchor.

An active anchor becomes dormant only after at least six strong absence frames
from at least three camera positions separated pairwise by 0.25 m. A strong
presence observation clears pending absence support. A dormant anchor becomes
active again after two consecutive strong presence observations. Suppression
is therefore multi-view, occlusion-safe, causal, and reversible.

## Integration

`src/oviv2/anchor_visibility.py` owns the pure policy, evidence extraction,
deterministic sampling, and state transition logic. It does not read files or
know about TESSE-CD.

`run_crove_ovimap_static_anchor.py` optionally loads a hash-bound policy and the
RGB-D export already bound by the anchor manifest. It evaluates all frames in
chronological order, adds synthetic unbound-anchor trajectory validity and
lifecycle transitions, and passes the dormant ID set to each checkpoint
composition. The composition core only omits explicitly dormant unbound
anchors; its default path remains unchanged.

The candidate uses a distinct `causal_visibility_candidate` readout role.
Visibility diagnostics, policy binding, RGB-D export binding, evidence counts,
and transitions are published in `runtime_diagnostics.json`, referenced by the
source index and propagated by the existing temporal exporter.

## Fail-Closed Rules

- Visibility is disabled unless an explicit policy file is supplied.
- The visibility role and policy must be supplied together.
- The policy cannot suppress a bound or unknown anchor ID.
- Dataset scene, frame count, frame IDs, and timestamps must match the source
  run exactly.
- Non-finite geometry, non-invertible poses, malformed policy values, missing
  RGB-D bindings, or input drift abort publication.
- Existing output directories are never overwritten.

## Promotion Gate

Only the single fixed Apartment configuration is evaluated. It may be frozen
for the one-shot Office run only if it improves the official Apartment Ghost
and object metrics, preserves current-map validity, passes causal export and
bridge validation, and keeps T1 exactness/non-interference green. No threshold
search is performed on Office.
