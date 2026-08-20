# CROVE State-Compatible Causal Evidence Routing Design

Date: 2026-08-20
Scope: TESSE-CD A4 temporal branch only
Selection split: Apartment only; Office remains held out
Public method name: CROVE
Stable internal artifact key: OVIV2

## Objective

Improve current-state dynamic mapping without changing the cumulative static
map. The optimization turns CROVE's existing lifecycle, identity, epoch, and
ownership mechanisms into one explicit rule: an observation may update only the
state components for which it is causally and observationally admissible.

The design closes three audited implementation gaps:

1. active dynamic-state updates currently consume geometric confidence only,
   even when the accepted association carries qualified identity evidence;
2. temporal export currently uses the fused submap centroid, which averages
   retained history instead of reporting the current observation;
3. active assignments update the persistent appearance prototype even when the
   association is not a qualified identity match.

This is narrower than claiming the first current-state, dynamic, or
open-vocabulary map. Prior work already establishes those task categories. The
paper contribution is the state-compatible causal evidence rule and its
measured effect under occlusion, reappearance, and same-class ambiguity.

## State Factorization

For identity `i` at frame `t`, CROVE maintains

```text
S_i^t = (I_i^t, E_i^t, P_i^t, D_i^t, G_i^t, O_i^t, R_i^t, M^cum_t)
```

where:

- `I` is persistent instance identity and its appearance prototype;
- `E` is existence/lifecycle belief;
- `P` is the current observed pose center;
- `D` is dynamic-state evidence with hysteresis;
- `G` is the current object-geometry epoch;
- `O` is reversible object/background ownership;
- `R` is the current-state temporal readout;
- `M^cum` is the cumulative static map used by T1.

The factorization is conceptual. The first implementation reuses existing
runtime and export state, so it adds no checkpoint or public artifact field.

## Evidence Types And Admissibility

Each evidence token has a type, current observable support, causal provenance,
and confidence. The admissibility relation is

```text
A(z, s) in {0, 1},
S_i^t[s] = F_s(S_i^(t-1)[s], {z_i^t | A(z_i^t, s) = 1}).
```

The closed matrix for the A4 temporal branch is:

| Evidence | I | E | P | D | G | O | R | M^cum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| observed present support | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| visible free-space absence | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| occluded / out-of-view / depth-unknown | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| accepted local geometry | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 0 |
| qualified appearance identity | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| current observation center | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| frozen static fusion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

Zero means that the evidence cannot modify the component. It may still advance
an observation timestamp used for deterministic replay. In particular:

- occlusion is neutral rather than negative existence evidence;
- visible free space may lower existence and release ownership, but may not
  rewrite identity or geometry;
- appearance may protect identity and support motion only after the existing
  high-confidence identity gate accepts the pair;
- current observation position may drive the temporal readout but never moves
  retained cumulative geometry;
- the temporal branch has no route to `M^cum`.

## Routed Active Motion

For an active accepted assignment, keep each displacement bound to the evidence
that produced it:

```text
d_g^t = ||p_geom_i^t - p_geom_i^(t-1)||_2,
d_a^t = ||c_obs_i^t - c_obs_i^(t-1)||_2.
```

The previous appearance endpoint is the A4 export tracker's last observed
center. It is therefore independent of the fused submap centroid. Let

```text
q_g = geometric motion confidence,
q_a = clipped appearance similarity,
G = motion decision is not rejected and q_g > 0,
Q = existing high-confidence identity match and q_a is finite,
M_g = G and d_g >= delta_motion and q_g >= tau_motion,
M_a = Q and d_a >= delta_motion and q_a >= tau_motion.
```

The routed observation qualifies as motion only when

```text
M_g or M_a.
```

If both tokens are admissible, the router selects a threshold-qualified token
before any non-qualified token and uses that token's own `(d, q)` pair. If no
token qualifies, it selects only among tokens whose own displacement reaches
the floor. It never combines `d_g` with `q_a` or `d_a` with `q_g`. This prevents
a strong appearance match from validating an incompatible weak geometric jump.

The existing consecutive-frame and static-off hysteresis remain unchanged.
Geometry-only behavior is unchanged. Identity evidence does not relax the
association gate, displacement floor, confidence threshold, or consecutive
motion requirement. A rejected geometric estimate can contribute only through
an independently qualified identity match.

Dormant re-identification remains the existing special case: it already binds
two temporally separated, identity-qualified endpoints and may mark a qualified
displacement dynamic without a consecutive-frame warm-up.

## Identity Prototype Isolation

An active assignment may update the persistent appearance prototype only when
the existing assignment diagnostic is a high-confidence identity match with a
finite appearance similarity. A geometry- or semantic-only assignment still
updates current existence, semantic evidence, and admissible local geometry,
but it cannot contaminate the persistent instance prototype.

New identities initialize their prototype from their first observation.
Dormant bank-only re-identification is already required to carry qualified
identity evidence and retains its current prototype update behavior. An
explicit feature-model mismatch keeps the existing compatibility behavior: it
atomically replaces the representation and provenance without blending
incomparable embeddings. It is not treated as motion evidence.

## Dual-Center Readout

The runtime keeps two centers without adding a schema field:

```text
c_map(i,t) = mean of retained fused submap world points,
c_obs(i,t) = current observation centroid in world coordinates.
```

For A4 observed entries, `TemporalExportSample.centroid_xyz` and
`TemporalExportTrackerEntry.last_centroid_xyz` use `c_obs`. The identity bank,
association target, retained geometry, static snapshot, A2/A3 exports, and T1
outputs continue to use `c_map`.

For an unobserved entry, the tracker retains its last observed center. Thus
missing observations cannot manufacture motion, while the next qualified
observation measures displacement from the last actual endpoint.

## Runtime Integration

Create a pure `temporal_evidence_router` module that owns:

- the evidence-to-component matrix;
- strict finite input validation;
- the active geometric/identity motion combination;
- identity-prototype update qualification;
- A4 current-center versus cumulative-center selection.

`temporal_runtime.py` remains the state-transition owner. It calls the router at
the three audited gaps and otherwise retains existing lifecycle, association,
epoch, ledger, identity-bank, and export behavior.

No new configuration threshold is introduced. Existing A4 thresholds remain
authoritative. A2 and A3 provide unchanged geometry-only controls. Formal
component ablations use fixed source variants after the full method is frozen;
the publication configuration does not expose a benchmark-specific switch.

## Safety Invariants

1. Runtime decisions consume only the current frame and prior runtime state.
2. Ground truth, evaluator output, future frames, and scene labels never enter
   routing decisions.
3. Occluded, out-of-view, and depth-unknown evidence is neutral for existence,
   identity, pose, motion, epoch, ownership, and readout.
4. Visible absence cannot update identity, pose, motion, or retained geometry.
5. Unqualified appearance cannot update the identity prototype or compensate
   for rejected geometry.
6. A single displaced active observation cannot satisfy the existing
   consecutive-motion requirement.
7. Current observation centers affect only the A4 temporal tracker/readout.
8. T1 protected files, cumulative geometry, and static export are byte-identical.
9. Prefix replay produces identical decisions and exports.
10. Missing or non-finite optional evidence fails closed to the geometry-only
    path rather than becoming high confidence.

## Focused Tests

The implementation must first add failing tests for:

1. every evidence type's exact admissible-component set;
2. geometric-only, identity-only, combined, and empty motion routes;
3. non-finite and malformed route inputs failing closed or raising at the public
   validation boundary;
4. two consecutive displaced active observations with qualified identity
   evidence becoming dynamic when geometry confidence is zero;
5. the same sequence without qualified identity evidence remaining on the old
   geometry-only path;
6. one displaced frame and sub-threshold jitter remaining static;
7. unqualified active association leaving the appearance prototype unchanged;
8. A4 export using the current observation center while retained geometry and
   the neutral snapshot remain unchanged;
9. A2/A3 preserving the fused-center export behavior;
10. dormant re-ID, rejected low-identity assignment, epoch purity, ledger
    reversibility, prefix replay, and T1 non-interference remaining green.

## Experiment Contract

Development and selection use Apartment only. After source, configuration, and
artifact identities are frozen, Office is evaluated once.

The full method is compared with fixed variants that remove exactly one route:

- no identity-prototype isolation;
- geometry-only active motion;
- fused-center temporal readout;
- untyped routing, where accepted evidence is allowed to update all legacy
  components;
- full state-compatible routing.

Report official Obj./Dyn./Chg. F1, common-v2 current mIoU, ghost rate,
background F@5cm, recovery latency, identity switches/fragmentation, epoch
resets, and qualified identity-route counts. Stress cases must include full
occlusion, partial visibility, same-class lookalikes, non-rigid motion, and
displaced reappearance.

The method advances to the held-out run only if:

- dynamic F1 or change F1 improves over the current A6 branch by at least
  `0.01` absolute;
- object F1 and current mIoU each regress by no more than `0.01` absolute;
- ghost rate increases by no more than `0.02` absolute, and identity-switch
  precision does not regress;
- T1 exactness, causal provenance, package, and source-binding gates pass.

Diagnostic values without checked-in source bindings remain development-only.
No superiority or SOTA statement is enabled until the formal registry contains
verified T2/T3/T4 cells.

## Paper Claim Boundary

The defensible claim is:

> CROVE factorizes persistent identity, existence, current pose, geometry
> epochs, ownership, and cumulative mapping, then admits each causal observation
> only to compatible state components. This prevents occlusion-driven removal,
> identity contamination, and history-biased current-state readout while leaving
> the cumulative static map unchanged.

The paper must not claim the first dynamic map, first current-state map, first
open-vocabulary temporal map, or first separation of visibility and identity.
Those broader claims conflict with prior work identified in the literature
audit.

## Rollback

The implementation adds no persistent schema migration. If Apartment formal
metrics or causal gates regress, revert the router integration and retain A6.
Existing artifacts remain readable because export and checkpoint schemas do not
change.
