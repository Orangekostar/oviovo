# CROVE Dual-Center Dynamic Readout Design

Date: 2026-08-17
Scope: TESSE-CD temporal A4 branch only
Selection split: Apartment only; Office remains held out

## Objective

Improve CROVE's TESSE-CD dynamic-object recall without changing the fused
static map used by T1. The change addresses two independent failure modes:

1. exported trajectory samples use the centroid of the fused submap, which
   lags or averages across object motion;
2. active tracks require geometric registration confidence even when the
   existing association already provides a high-confidence identity match.

The method therefore separates the center used for static map fusion from the
center used for temporal readout, and admits identity-qualified displacement as
an additional dynamic-motion evidence source.

## Evidence And Diagnosis

The A6 Apartment artifact shows that dynamic classification and localization
must both improve:

- official dynamic F1 remains approximately 0.01 despite improved object F1;
- at two inspected dynamic timestamps, the fused-submap centroid is 1.288 m
  and 1.424 m from the target, while the current object-pose center is 0.559 m
  and 0.266 m away;
- a diagnostic substitution of current object-pose centers increases simulated
  true positives from 996 to 1489, including a checkpoint increase from 2 to
  16, but does not by itself make dynamic F1 competitive.

These values are diagnostic development evidence, not publication results.
They establish that the static fused centroid is the wrong temporal readout
coordinate and that a second classification change is still required.

## Design

### 1. Dual centers

Each active entity has two conceptually distinct centers:

```text
c_map(i,t) = mean of the retained fused submap points
c_obs(i,t) = translation of the current per-frame pose estimate
```

`c_map` remains the source for retained geometry, static snapshots, map
association, identity-bank state, and all T1 outputs. For an active assignment,
`c_obs` is taken from the current motion result before the geometry epoch decides
whether to retain or integrate that pose. For a new or dormant-reidentified
entity, it is the current observation centroid. `c_obs` becomes the center of an
observed `TemporalExportSample` in the TESSE-CD temporal A4 path.

This separation changes only the trajectory readout. It does not move fused
points, rewrite epochs, alter static export colors, or change the neutral map.
An unobserved tracker entry retains its previous stored center as before. The
alternate readout center is guarded by `ExecutionProfile.A4`; A2 and A3 exports
retain their current behavior.

### 2. Identity-qualified active motion

For an accepted active association between observation `o_t` and entity `i`,
define

```text
d_i(t) = ||p_i(t) - p_i(t-1)||_2
q_i(t) = clip(cosine(f_o(t), f_i(t-1)), 0, 1)
g_i(t) = existing geometric motion confidence
```

where `p_i` is the translation of the estimated object-to-world pose. The
appearance term is eligible only when the existing association diagnostic marks
the pair as `high_confidence_identity_match` and provides a finite appearance
similarity. Semantic-only and geometry-only matches do not qualify.

The evidence confidence is

```text
G_i(t) = motion decision is not rejected and g_i(t) > 0
Q_i(t) = the existing high-confidence identity gate qualifies
e_i(t) = max(g_i(t) if G_i(t), q_i(t) if Q_i(t))
```

An active observation advances the existing consecutive-motion state machine
when either `G_i(t)` or `Q_i(t)` is true, `d_i(t)` is at least the existing
displacement floor, and `e_i(t)` reaches the existing minimum motion confidence.
An ICP-rejected result is therefore still eligible only through an independently
qualified identity match. The current `minimum_consecutive_motion_frames`
requirement remains unchanged. Thus appearance evidence can recover non-rigid
or weak-ICP motion, but cannot turn a one-frame jump or low-displacement jitter
into a dynamic trajectory.

The dormant re-identification behavior introduced by A6 remains unchanged:
dormant re-ID already connects two identity-qualified temporal endpoints and
may assign dynamic state immediately when its existing displacement and
confidence thresholds pass. The active identity-evidence combination is also
guarded by `ExecutionProfile.A4`.

### 3. Geometry-epoch behavior

The combined evidence controls the same epoch transition that geometric motion
controls today:

- before the consecutive-motion threshold, retain the current epoch and avoid
  contaminating it with a large weak-motion observation;
- when the state first becomes dynamic, start a new geometry epoch at the
  current pose;
- while already dynamic, integrate through the current dynamic epoch;
- below the displacement floor, use the existing stationary integration path.

Rejected motion keeps the existing forced-new-observation and identity-gated
epoch-reset behavior. No association gate is relaxed.

## Causality And Safety Invariants

1. Runtime decisions consume only current observations and prior state.
2. Ground truth, evaluator outputs, future frames, and scene-specific labels
   never enter runtime configuration or decisions.
3. T1 protected sources, cumulative-map geometry, static snapshots, and frozen
   T1 configuration remain byte-for-byte unchanged.
4. Missing, non-finite, or non-qualified appearance evidence falls back to the
   existing geometric-confidence path.
5. The existing displacement floor, confidence threshold, consecutive-frame
   threshold, and association distance bounds remain authoritative.
6. No public artifact schema or checkpoint schema changes are required.
7. Apartment is the only development split. Office is evaluated once after the
   complete configuration and source identity are frozen.

## Alternatives Rejected

### Threshold-only tuning

Lowering the geometric confidence threshold cannot correct the exported center
and increases sensitivity to static registration noise. It does not address the
two observed failure modes.

### Bridge-side trajectory filtering

Dropping static samples only when the current frame is dynamic may reduce false
positives, but it changes trajectory semantics at evaluation time and discards
valid history. The correction belongs in the causal runtime state and readout.

### Semantic class priors

Treating selected categories as dynamic is dataset-specific, fails for moved
furniture, and risks using benchmark knowledge as a runtime shortcut.

## Implementation Boundary

The production change is confined to `src/oviv2/temporal_runtime.py` and focused
temporal runtime/export tests. It will:

1. derive the observed export center from the current object pose;
2. combine qualified appearance confidence with geometric motion confidence in
   the active-association branch;
3. keep dormant re-ID, association, identity-bank, static-map, and T1 code paths
   otherwise unchanged.

No configuration field is added. Localization-only and evidence-only ablations
are produced from fixed source variants during development rather than adding a
runtime switch to the publication configuration.

## Test Plan

Add focused regression tests before implementation:

1. two or more consecutive displaced active observations with qualified
   identity evidence become dynamic when geometric confidence is zero;
2. the same sequence without qualified appearance evidence stays on the
   geometric-only path;
3. one displaced frame does not satisfy the consecutive-motion requirement;
4. high-confidence static jitter below the displacement floor remains static;
5. an observed export sample uses the current pose center while retained fused
   points and the neutral snapshot remain unchanged;
6. dormant reappearance behavior and the existing rejected-motion
   forced-new/epoch-reset decisions remain unchanged;
7. prefix replay produces identical decisions and exports, demonstrating
   causality and determinism;
8. the T1 protected-source verifier and T1 non-interference suite remain green.

Run focused unit tests first, then the temporal runtime/export suite, provenance
and package gates, and finally the frozen T1 verification commands.

## Experiment And Decision Gate

Use the existing preregistered TESSE-CD process:

1. evaluate the full dual-center plus dual-evidence method on Apartment;
2. compare with A6 and fixed localization-only/evidence-only ablations;
3. inspect official Obj./Dyn./Chg. F1, common-v2 current mIoU, ghost rate,
   background F@5cm, recovery latency, and identity diagnostics;
4. select only through the existing formal gate, then freeze code, config,
   source bindings, and artifacts before the one-shot Office evaluation.

The method advances only if official dynamic performance improves over A6 while
the preregistered object, change, common-v2, provenance, and T1 gates pass. A
diagnostic proxy or visually favorable trajectory is not sufficient evidence.

## Failure Modes And Rollback

The main risk is a high-confidence appearance match between repeated objects.
The existing active association bounds, displacement floor, and consecutive
motion requirement limit this risk. Report qualified-match counts,
displacement/confidence distributions, epoch resets, and per-event precision.

If static false positives increase or formal gates regress, revert the temporal
runtime commit and retain A6. The design makes no persistent schema migration,
so rollback requires no artifact conversion.
