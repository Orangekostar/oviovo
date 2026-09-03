# CROVE P6 Localized Current Ownership Design

Date: 2026-09-03  
Base: `dd3247f2d9e81e31a7508b8b692480dbb8855246`  
Branch: `research/crove-localized-current-ownership`

## Problem

P5 proves that causal free-space evidence can reduce stale-map Ghost, but its
whole-anchor binary decision removes all dense OVI-MAP geometry at once. The
Apartment result reduces Ghost from `0.646883` to `0.44167745295296196` and
raises current mIoU from `0.142897` to `0.14956262241133872`, while Object F1
drops from `0.372762` to `0.3679520027114184`. P6 must recover the missing
`0.004809997288581558` Object F1 without surrendering P5's current-state gains.

The method change is limited to readout ownership. The immutable causal
OVI-MAP anchor remains the geometry authority; CROVE remains the temporal state
authority.

## Measured Evidence

- P2 attributes `1,592,890 / 2,152,474 = 74.002752%` of official object Ghost
  matches to unbound OVI-MAP anchor geometry.
- P5 emits ten active-to-dormant unbound-anchor transitions. Five occur at
  frames 355-356 and five at frames 636 and 1214-1218.
- The ten transitions include the five P2 dominant stale anchors, but P5's
  Object F1 regression shows that whole-object removal also deletes useful
  geometry.
- The 36 unbound Apartment anchors contain 41,797 unique 5 cm voxels. P5 samples
  15,938 anchor voxels (15,514 unique world keys) for visibility evidence.
- A frozen-RGB-D replay over frames 263-1744 recorded 3,891,358 direct
  PRESENT/ABSENT observations. The direct 5 cm PRESENT-to-ABSENT or
  ABSENT-to-PRESENT flip rate is `0.015130450603619611`.
- At 10 cm and 20 cm aggregation, blocks containing simultaneous PRESENT and
  ABSENT evidence occur at rates `0.04337290612402827` and
  `0.08955982676960067`. Coarser blocks therefore create more contradictory
  local evidence.
- Raw array-state estimates, including voxel keys and three stored absence
  viewpoints, are 2,382,429 bytes at 5 cm, 704,178 bytes at 10 cm, and 205,086
  bytes at 20 cm. The 5 cm state is small relative to the 18.8 MB immutable
  anchor snapshot and avoids block conflicts.
- P4A shows unconditional dense centroid translation is unsafe. For example,
  `ovimap:2` has zero dense-to-compact coverage at 5 cm and median nearest
  neighbor errors of 1.41-2.05 m, whereas well-aligned cases such as
  `ovimap:131` and `ovimap:85` reach at least 0.96 coverage with sub-4 cm p90
  error.
- Official Dyn aggregates are `188` detected, `19,556` missed, and `22,431`
  hallucinated samples. The bridge binds prediction entity IDs to Khronos node
  symbols, but `dynamic_objects.csv` does not expose per-sample associations.

## Code-Path Binding

The P6 implementation changes only these causal readout paths:

1. `src/oviv2/anchor_visibility.py` classifies frozen anchor voxels and advances
   their reversible ownership state.
2. `scripts/evaluation/run_crove_ovimap_static_anchor.py` replays RGB-D frames
   in increasing order, captures ownership only at official checkpoints, and
   writes hash-bound mask sidecars.
3. `src/oviv2/ovimap_static_anchor.py` filters unbound dense anchor points by a
   typed ownership object and selects dense moved geometry only through an
   explicit agreement result.
4. `src/evaluation/crove_anchor_counterfactual.py` analyzes P5 suppression
   variants without becoming a runtime rule.
5. The Dyn provenance sidecar reads existing official CSV/visualization,
   bridge, composition, temporal, and runtime-attribution artifacts. It never
   changes official metrics.

T1 mapper, frontend, vocabulary, anchor construction, association, motion,
lifecycle, and official evaluator code remain unchanged.

## Counterfactual Protocol

P6-A derives the transitioned anchor set from the P5 diagnostics and ranks its
P2 Ghost contribution from the hash-bound P2 attribution artifact. IDs are
diagnostic identities only.

The immutable variant set is:

- `CF0`: no P5 suppression;
- `CF1`: all ten P5 suppressions;
- `CF2_<token>`: CF1 except one transitioned anchor;
- `CF3_<token>`: only one transitioned anchor;
- `G1`: the five transitioned anchors with greatest P2 Ghost mass;
- `G2`: the other five transitions.

Each variant filters the same causal P5 visibility timeline and recomposes the
same source checkpoints. It does not rerun the source mapper or alter evidence
timing. Counterfactual composition is guarded as a diagnostic-only readout and
cannot satisfy a production promotion contract.

For each anchor, the analyzer records transition frame, label, dense point and
sampled-voxel counts, evidence counts, support and distinct views at transition,
first absence frame, suppression latency, and bounding-box extent. Each fully
evaluated variant records Object/Dyn/Change F1, current mIoU, and Ghost, plus
deltas against CF0 and CF1. Missing official inputs are reported as unavailable,
never estimated.

P6-A is GO only if measured features provide a plausible ID-independent
distinction between Ghost benefit and Object damage. Regardless of that result,
the final P6 policy remains spatial evidence based.

## State Model

`AnchorCurrentOwnership` is a typed, validated snapshot for one unbound anchor.
It contains:

- sorted unique full 5 cm anchor voxel keys;
- a deterministic evidence-sample index;
- current/suppressed bits;
- per-voxel absence observation counts;
- up to the configured minimum number of distinct absence viewpoints;
- per-voxel PRESENT streaks and first-absence frame;
- last consumed frame and timestamp;
- the exact policy identity and voxel size.

Numeric arrays are copied on advance and exposed read-only. The immutable
OVI-MAP points are never edited in place. Unsampled full-state voxels start
current and remain neutral unless selected by the deterministic P5 sampler.

`whole_anchor_status` is derived, not independently updated:

- `unchanged`: every state voxel is current;
- `partially_suppressed`: both current and suppressed voxels exist;
- `dormant`: no current voxel exists.

This avoids divergent object-level and voxel-level truth.

## Visibility Semantics

Projection retains P5's 5 cm quantization, 10 cm depth tolerance, 10 m maximum
depth, 1,000-voxel evidence cap, six absence observations, three separated
viewpoints at 25 cm baseline, and two-frame PRESENT restoration.

For each sampled local unit:

- PRESENT is positive evidence. While current it clears pending absence
  support; while suppressed it advances restoration.
- VISIBLE_ABSENT is negative evidence. It advances absence support and can
  suppress only that voxel.
- OCCLUDED is neutral and preserves accumulated absence support.
- UNOBSERVED and invalid depth are neutral and preserve accumulated absence
  support.
- Any non-PRESENT observation breaks a pending restoration streak.

P5's `minimum_tested_voxels` and anchor-level PRESENT/ABSENT fraction fields
remain frozen in the policy receipt and legacy whole-anchor compatibility path.
They are spatial aggregation rules, so they are not applied to a single 5 cm
unit. This is the sole L1 semantic change: aggregation moves from the whole
anchor to the projected voxel.

## Memory Bound

The state allocates at most all unique 5 cm voxels of the 36 unbound anchors.
For the frozen Apartment anchor this is 41,797 units and approximately 2.38 MB
of raw state arrays. Evidence projection remains capped at 1,000 voxels per
anchor. Per-checkpoint masks are bit-packed and stored as separate canonical,
hash-bound artifacts; entity JSON stores only counts and policy identity.

The measured 5 cm instability is lower than the measured within-block conflict
of both tested coarser choices. Therefore L1 uses per-voxel state and L2 is
preregistered as `NOT_RUN_NOT_NEEDED`. It may be activated only if the
implemented memory measurement exceeds 8 MB or deterministic replay exposes a
state inconsistency, not because of official metrics.

## Readout Contract

An unbound dense anchor point is emitted exactly when
`floor(point / 0.05)` is current in its `AnchorCurrentOwnership`. Filtering
never replaces a point by the voxel center and never downsamples native OVI-MAP
geometry.

Partial output preserves entity ID, semantic label, embedding, score, and
lifecycle. Metadata records `state_authority=crove_anchor_visibility`,
`overlay_state`, active/suppressed voxel and point counts, and policy ID. Large
masks live in a separate sidecar bound by SHA256 and byte count.

The legacy `suppressed_unbound_anchor_ids` argument remains valid and unchanged
for P5 reproduction. A call must use either that legacy whole-anchor input or
typed localized ownership, never both.

## Causality

- State starts from the causal anchor cutoff at frame 262.
- Frame `t` ownership uses only RGB-D, pose, and timestamps through `t`.
- Checkpoint `t` stores a copy of ownership after consuming frame `t`.
- No ground-truth change, target, metric, future frame, or Office observation is
  a runtime input.
- Counterfactual variants reuse the same P5 causal transition timeline; IDs
  select diagnostic variants only after the causal evidence has been produced.
- Byte-identical replay and a future-frame noninterference test are mandatory.

## Reversibility

Suppression modifies only sidecar state. Two consecutive later PRESENT
observations restore the affected voxel and its original dense anchor points.
OCCLUDED, UNOBSERVED, and invalid-depth frames cannot suppress or restore a
voxel. Restoration clears absence counts, saved viewpoints, and first-absence
frame for that voxel.

## Dense-Readout Gate

P6-D is evaluated standalone before combination. For each moved bound object it
compares the translated dense anchor template with the exact compact temporal
geometry using:

- compact-to-template and template-to-compact coverage at 5 cm;
- median and p90 bidirectional nearest-neighbor distance;
- centroid residual;
- per-axis bounding-box extent residual and ratio where defined.

The frozen gate is derived from the 5 cm state scale:

- both directional coverage at 5 cm must be at least 0.90;
- bidirectional median must be at most 0.05 m;
- bidirectional p90 must be at most 0.10 m;
- centroid residual must be at most 0.05 m;
- every axis must have absolute extent residual at most 0.10 m; axes with both
  extents at least 0.05 m must also have ratio in `[0.5, 2.0]`.

These thresholds are fixed before P6-D official evaluation and are not fit to
known object IDs. A failed gate emits the exact `temporal_compact` prediction,
including point ordering and metadata except for additive audit fields.

## Dyn Provenance Plan

P6-E binds, in order:

1. official row `(MapName, QueryTime)` to one bridge checkpoint;
2. bridge `node_symbol` to prediction entity ID;
3. prediction to composition authority, anchor/temporal ID, overlay state,
   semantic state, geometry epoch, and readout validity;
4. temporal ID and frame to runtime association and motion diagnostics where a
   captured record exists.

Static evaluator visualization associations may provide exact per-node
provenance. Official dynamic CSV rows expose only aggregate counts; dynamic
mass is `exact` only when a deterministic replay from hash-bound trajectory
inputs reproduces the official row and binds one prediction/error event.
Otherwise it is `ambiguous`; absent required evidence is `unavailable`.
Ambiguous mass is never redistributed heuristically.

The report totals exact, ambiguous, and unavailable official Dyn mass and
partitions supported records into proposal, association, motion, geometry,
identity, lifecycle, and readout mechanism buckets. Proposal, association,
motion, or ReScene recommendations require dominant exact evidence.

## Promotion Gates

Apartment thresholds are frozen before candidate execution:

| Metric | P6 hard gate |
| --- | ---: |
| Object F1 | `>= 0.372762` |
| Dynamic F1 | `>= 0.06922505723328032` |
| Change F1 | `>= 0.0880875002614133` |
| current mIoU | `>= 0.14956262241133872` |
| Ghost | `<= 0.44167745295296196` |
| source frame coverage | `1745 / 1745` |
| official state coverage | `43 / 43` |
| T1 exactness/noninterference | all green |

The ablation order is A6, c553 static anchor, P5 whole-anchor visibility, P6
localized visibility, P4A dense translation, and P6 hybrid dense readout.
Localized plus hybrid is run only after both standalone mechanisms have valid
measurements.

Office remains `NOT_RUN_HELD_OUT` unless one complete Apartment configuration
passes every hard gate. A passing candidate freezes commit, config, artifact
hashes, and identity before one Office run.

## Rejected Alternatives

- Manual anchor-ID filtering: diagnostic IDs do not generalize and cannot be a
  final runtime rule.
- Blind P5 threshold sweep: it confounds localization with threshold fitting.
- Global 5 cm to 1 cm state-grid change: it changes memory, evidence density,
  and the evaluation-scale contract simultaneously.
- 10 cm or 20 cm block ownership: measured mixed PRESENT/ABSENT conflicts are
  2.9x and 5.9x the direct 5 cm flip rate.
- Hybrid block hysteresis for L1: the 5 cm state is only about 2.38 MB and does
  not meet the preregistered reason to coarsen.
- Unconditional dense centroid translation: P4A contains zero-coverage,
  meter-scale failures.
- Dense template fallback after a failed gate: failure must be byte-equivalent
  to compact geometry.
- ReScene before Dyn provenance: identity architecture must not be changed from
  aggregate Dyn counts alone.
- Estimated or fabricated official counterfactual metrics: unavailable values
  remain explicitly unavailable until the official evaluator runs.

