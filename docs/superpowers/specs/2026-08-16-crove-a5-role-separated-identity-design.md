# CROVE A5 Role-Separated Identity Design

## Goal

Improve TESSE-CD object, dynamic, change, ghost, and revealed-background
metrics without changing the frozen T1 cumulative/static path.

## Observed Failure

The recovered A4 diagnostic produces 200 identities, including 47 one-frame
tracks, 48 identities labeled dynamic, 469 geometry epochs, and 1556 lifecycle
transitions. Its current mIoU remains usable, but ghost rate and background
recovery regress, while dynamic and change F1 remain far below Khronos.

The immediate code-level cause is role leakage in temporal association. A
recent change applies the dormant re-identification radius to ACTIVE and
UNCERTAIN targets whenever appearance evidence passes the re-ID threshold.
With repeated categories such as chairs and sofas, similar appearance can then
override local geometry and merge different objects across meters. This also
violates the existing A4 contract that only dormant candidates receive the
wider spatial gate.

## Decision Hypotheses

For observation `o_t` and identity `e`, CROVE evaluates three distinct roles:

1. active continuation `H_A`;
2. dormant reactivation `H_D`; and
3. persistent birth `H_B`.

They must not share an admission rule.

### Active continuation

An ACTIVE or UNCERTAIN edge is eligible only when

`d(c(o_t), c(e)) <= r_A`.

Its score may use visual, semantic, size, and geometry evidence, but appearance
cannot widen `r_A`.

### Dormant reactivation

A DORMANT edge is eligible only when

`d(c(o_t), c(e)) <= r_D`, `r_D > r_A`,

and appearance provenance, appearance similarity, and semantic compatibility
all pass the re-identification qualification. Only this role records a re-ID
opportunity or trigger.

### Persistent birth

An unmatched proposal remains tracker-local until it receives at least two
consistent hits. Only accepted tracks allocate a persistent identity, object
submap, geometry epoch, and current-map ownership.

### Dynamic evidence

Motion is promoted to dynamic state only after repeated accepted motion with
displacement above the sensor/association noise floor and sufficient motion
confidence. The first A5 diagnostic uses existing state machinery with more
conservative values; a new estimator is out of scope until this minimal causal
change is measured.

## A5 Development Configuration

- execution profile: A4;
- `confirm_hits = 2`;
- active/uncertain gate: existing local `maximum_centroid_distance_m` only;
- dormant gate: existing `maximum_reid_distance_m` after qualification;
- `minimum_consecutive_motion_frames = 3`;
- `displacement_floor_m = 0.15`;
- `minimum_motion_confidence = 0.8`;
- all proposal, lifecycle, geometry, background, and static frontend settings
  unchanged from A4.

These values are a single preregistered development point, not a sweep selected
after evaluation.

## Alternatives Rejected

- Threshold-only tuning: fast, but it does not correct role leakage and has a
  weak scientific mechanism.
- Dense cross-session semantic correspondence: high implementation cost and
  direct novelty overlap with OASIS-Map.
- Looser background release: unsafe before identity masking is corrected and
  can increase false removal under occlusion.

## Validation

1. Unit tests prove active targets never inherit dormant distance while
   dormant reactivation remains available.
2. T1 protected-source gate and non-interference tests remain exact.
3. Run one Apartment A5 diagnostic with the causal schedule.
4. Promote only if object F1 and current mIoU are non-inferior to A0 and the
   strict-improvement axes show a credible joint gain.
5. If promoted, bind the configuration and transfer exactly once to Office.

Recovered proxy metrics are diagnostics only and cannot be submitted.
