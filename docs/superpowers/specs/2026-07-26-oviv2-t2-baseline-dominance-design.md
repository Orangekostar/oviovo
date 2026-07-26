# OVIV2 T2 Baseline Dominance with Frozen T1

Date: 2026-07-26

Status: approved design

## Objective

Improve OVIV2 on TESSE-CD T2 until every claimed metric exceeds the strongest
non-oracle baseline under the same protocol, while preserving the frozen T1
cumulative-map artifacts exactly.

The final two-scene targets are:

- Dynamic F1 greater than 0.822.
- Change F1 greater than 0.302.
- Ghost rate lower than 0.218.
- Background F-score at 5 cm greater than 0.163.
- Recovery latency lower than 450 frames.
- Current-map mIoU at least 0.200.
- Object F1 at least 0.469.

These are promotion targets, not claimed results. A value may enter the paper only
after it is produced by the frozen evaluation package.

## Scope

This design may change only the OVIV2 method-side temporal readout, causal temporal
export, and their tests and packaging. It must not change:

- TESSE-CD GT DSG, background, or change annotations.
- The causal checkpoint schedule or query timestamps.
- The official Khronos evaluator, its configuration, or its thresholds.
- The common-v2 evaluator or metric definitions.
- The occlusion target manifest, 5 cm world-voxel mapping, or unique-winner rule.
- Baseline artifacts or baseline result values.
- The frozen T1 cumulative mapping algorithm or its inputs.

The method must not use GT classes, GT motion, GT trajectories, future frames, or
evaluation outcomes to construct predictions.

## Evidence Behind the Redesign

The released OVIV2 result is not uniformly weak. Object F1 and current-map mIoU are
already 0.469 and 0.200, while Dynamic F1, Change F1, Ghost, and background recovery
are the limiting columns.

The completed A0-A3 development runs expose two distinct failures:

| Profile | Current mIoU | Ghost | BG F5 | Anchor coverage | Occluded retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.19935 | 0.57081 | 0.06419 | 4/66 | 10/17978 |
| A1 | 0.19455 | 0.57128 | 0.06419 | 4/66 | 10/17978 |
| A2 | 0.12248 | 0.66959 | 0.06419 | 4/66 | 160/17978 |
| A3 | 0.12248 | 0.66959 | 0.05235 | 4/66 | 160/17978 |

A4 completed its recovered Apartment mapping run but has no common, occlusion,
official, packaged, or selected result. No A0-A4 result is therefore a final T2
result.

The dynamic export path is also structurally wrong:

- Entity centroids are recorded only at sparse evaluation checkpoints.
- The bridge labels them as native per-frame trajectories.
- Two trajectory samples are treated as sufficient evidence that an entity is
  dynamic.
- In A0, 209 of 226 assignments are consequently marked dynamic, including many
  static objects.
- The official evaluator performs exact timestamp matching with a 0.5 m centroid
  threshold. The resulting aggregate is TP=7, FN=19737, and FP=106899, despite an
  object F1 of 0.4735.

The 4/66 anchor coverage is a separate method-side problem. Most failures have zero
world-voxel overlap, not ambiguous evaluator matching. This means no lifecycle or
threshold search can recover the missing instances without improving temporal
proposal and ownership recall.

## Alternatives Considered

### Threshold search only

This is inexpensive but rejected. A2 and A3 already regress mIoU, Ghost, and
background F5. Their failure is structural rather than a local threshold choice.

### Learned temporal change network

This may have a high ceiling but is deferred. It needs reliable temporal labels,
introduces training and transfer risk, and increases T4 cost. The current failures
can first be addressed using causal geometric and lifecycle evidence already
available to the method.

### Diagnostic-first dual readout

This is the selected approach. It corrects the method-side causal export, repairs
temporal proposal ownership, separates identity from current geometry, and makes
background ownership reversible. The cumulative T1 path remains independent and is
verified by exact artifacts rather than metric tolerance.

## Architecture

```text
Frame + cached observations
  +-> frozen Oviv2Runtime cumulative readout -> T1 cumulative audit artifact
  |
  `-> TemporalProposalRecovery
        -> IdentityMemory
        -> GeometryEpoch
        -> LifecycleBelief and readout eligibility
        -> ReversibleBackgroundLedger
        -> CurrentMapSnapshot
        -> CausalTemporalExport
        -> frozen T2 evaluators
```

### Frozen cumulative readout

The cumulative branch remains an `Oviv2Runtime` driven only by the frozen runtime
configuration and the original frame, observation, and dense-semantic inputs. No
temporal parameter, mask, association, motion result, lifecycle state, geometry
epoch, or background contribution may flow back into it.

All transitive cumulative dependencies are read-only in this work. New proposal,
association, motion, lifecycle, and background behavior must live in temporal-only
modules or wrappers rather than changing a module imported by the cumulative path.

Every temporal profile produces an unpublished cumulative audit artifact. The
artifact must match the frozen v1 reference inventory and SHA-256 values exactly.

### Temporal proposal recovery

The temporal branch combines existing tracked observations with object-class dense
semantic support. Unowned foreground support is grouped into bounded 3D connected
components and may form or extend a temporal proposal when it passes the configured
minimum support, depth consistency, and semantic confidence checks.

Proposal recovery is causal and method-only. It cannot inspect occlusion targets,
GT identities, GT motion, or future observations. It is introduced in A2 and is
reported separately as the A2-without-recovery micro-ablation.

The mechanism has one primary diagnostic: anchor mapping coverage. Full-scene T2
evaluation is blocked until Apartment development coverage is at least 80 percent.

### Identity memory

`IdentityMemory` stores stable identity evidence only:

- Temporal entity ID.
- Semantic posterior.
- Appearance prototype and feature-model identity.
- Motion prior and uncertainty.
- First and last observation times.
- Lifecycle belief.

It does not own current geometry. Capacity pressure evicts reclaimable geometry
before identity evidence. Identity is removed only under an explicit bounded-bank
policy recorded in diagnostics.

A2 and A3 use identity memory for active and uncertain entities but do not match new
observations against dormant identities. A4 additionally retains and searches the
dormant identity bank. A1 remains keyed by the frozen cumulative registry IDs.

### Geometry epoch

`GeometryEpoch` stores a versioned current-state hypothesis:

- Temporal entity ID and monotonically increasing epoch ID.
- Object-to-world pose.
- Object-local submap.
- Last accepted observation.
- `readout_valid` flag.
- Motion-decision provenance.

A reappearing identity may reuse an epoch only when association and motion checks
both accept it. A large displacement or rejected registration preserves the
identity but creates a new geometry epoch. Old and new geometry must never be fused
into the same epoch after motion rejection.

### Lifecycle and readout eligibility

Lifecycle belief and current-map visibility are separate state variables:

| Evidence | Identity lifecycle | Geometry readout |
| --- | --- | --- |
| PRESENT | update toward active | valid after accepted geometry update |
| OCCLUDED | neutral | retain previous valid epoch |
| OUT_OF_VIEW | neutral | retain previous valid epoch |
| DEPTH_UNKNOWN | neutral | retain previous valid epoch |
| VISIBLE_ABSENT | update toward dormant | invalidate old epoch immediately |

Visible absence must satisfy the configured signed-depth pixel count, fraction,
view diversity, and confidence checks. Occlusion can never increment an absence
streak or invalidate geometry.

The slow lifecycle hysteresis protects identity from a single false negative. The
immediate geometry invalidation prevents that hysteresis from forcing stale geometry
into the current map.

### Motion decision

Motion estimation returns exactly one of:

- `ICP_ACCEPTED`.
- `TRANSLATION_ACCEPTED`.
- `REJECTED`.

`REJECTED` forbids integration into the old submap. A low-confidence re-ID remains
dormant and the observation creates a new identity. A high-confidence re-ID with a
rejected old-epoch motion estimate retains the identity but starts a new epoch.

The dynamic-state estimator uses accepted causal motion and its uncertainty. It
must not infer dynamic state from semantic class or trajectory sample count. Its
consecutive-evidence count, displacement floor, and hysteresis are explicit bounded
configuration fields frozen before a full Apartment run.

The Apartment development search is limited to consecutive accepted-motion counts
of 2, 3, or 4; displacement floors of 0.05, 0.10, or 0.15 m; minimum motion
confidence of 0.6, 0.7, or 0.8; and static-off streaks of 5, 10, or 20 frames. The
selected tuple is frozen before Office and is never chosen from GT class identity.

### Reversible background ledger

The current masked TSDF prevents some future contamination but cannot remove a
previously fused dynamic surface. A3 replaces it with block-scoped contribution
ownership:

- Measurements near active or uncertain object support are not committed as
  background.
- Background revealed by reliable visible absence is first stored as provisional
  evidence.
- Consistent evidence from the configured number of frames and distinct views is
  committed.
- Each committed block retains enough source contribution metadata to rebuild the
  block without a contribution later attributed to a dynamic geometry epoch.
- If journal capacity or rebuild validation fails, the block rejects the new
  contribution rather than publishing an irreversible update.

Provisional evidence may be discarded under capacity pressure. Committed evidence
may not be silently discarded or overwritten.

The Apartment development search permits commit support of 2, 3, or 4 frames and 2
or 3 distinct view bins. A candidate that cannot retain the contribution journal
within the frozen T4 memory bound is rejected rather than given a larger budget.

### Causal temporal export

The exporter emits method-side evidence at every input frame:

- Stable temporal entity ID.
- Native timestamp and frame index.
- Current accepted centroid and observation count.
- Explicit dynamic state and motion confidence.
- Lifecycle transition events.
- Geometry epoch ID and readout validity.

For A0 and A1, a read-only exporter tracker derives kinematic evidence from the
frozen cumulative entity IDs and their per-frame native centroids. It cannot mutate
the cumulative runtime. For A2-A4, the exporter consumes the explicit temporal
motion decision and geometry epoch. Both paths use the same dynamic-state contract
and emit one native-time record per observed entity per frame.

At query time, the bridge uses only the prefix whose timestamps are no later than
the query. It exports trajectories only for entities classified as dynamic by the
method. It derives presence intervals from lifecycle transitions, not from gaps
between sparse evaluation checkpoints.

The bridge may adapt the method output to the unchanged Khronos input schema. It may
not interpolate GT timestamps, apply GT labels, infer dynamic state from sample
count, or alter evaluator thresholds.

## Per-Frame Transaction

For each frame:

1. Fingerprint all shared frame, observation, and dense-semantic inputs.
2. Advance the cumulative runtime using the frozen path.
3. Recover temporal proposals from method-visible evidence.
4. Associate proposals jointly against active, uncertain, and eligible dormant
   identity candidates.
5. Produce an explicit motion decision before integrating geometry.
6. Update lifecycle belief and geometry readout validity.
7. Stage object geometry, provisional background, committed-block rebuilds, and
   per-frame export records.
8. Validate scene, frame, timestamp, revision, finite values, ownership, and bounded
   capacities.
9. Re-fingerprint shared inputs and verify the cumulative deep fingerprint.
10. Publish both runtime states only if every check passes.

Any temporal failure restores the pre-frame cumulative and temporal states and
publishes neither. A rollback failure is fatal and cannot produce an evaluation
artifact.

## Execution Profiles

The existing A0-A4 meanings remain fixed:

| Profile | Increment |
| --- | --- |
| A0 | Frozen cumulative reference current readout plus corrected causal exporter |
| A1 | A0 plus signed visibility and probabilistic lifecycle overlay |
| A2 | A1 plus temporal proposal recovery, local object submaps, and translation-only motion |
| A3 | A2 plus reversible background ownership and reclaim |
| A4 | A3 plus dormant identity-bank re-ID and gated ICP with explicit rejection |

Each profile is an explicit execution profile, not a free combination of booleans.
The manifest component declaration must exactly match the derived profile or the run
fails before processing frames.

The following micro-ablations are permitted without changing the main profile table:

- A2 without proposal recovery.
- A3 with masking only and no reversible ledger.
- A4 without dormant candidates.
- A4 translation-only without ICP.

## Failure Handling

- Motion rejection never mutates an existing geometry epoch.
- Re-ID uncertainty preserves dormant identity and creates a new identity for the
  observation.
- Background journal overflow rejects new provisional evidence.
- A block rebuild mismatch keeps the previous committed block.
- Export timestamp, identity, or causality violations abort artifact publication.
- Empty opportunities are reported as unavailable, never converted to a zero or a
  positive claim.
- Two known absent lifecycles without a post-lifecycle checkpoint remain explicitly
  unavailable under the frozen schedule.

If A4 fails any T2 or T4 gate, selection falls back to A3. If reversible background
fails, selection falls back to A2. Metrics from different profiles may not be mixed
into a synthetic row.

## Verification

### Unit and property tests

- Occlusion never increments absence evidence or hides valid geometry.
- Reliable visible absence immediately hides geometry while retaining identity.
- Reappearance after large displacement starts a new geometry epoch.
- `REJECTED` motion cannot call object-submap integration.
- Background block commit, removal, and rebuild are deterministic and reversible.
- Journal overflow leaves the committed block byte-identical.
- Dynamic state depends only on causal method motion evidence.
- Per-frame trajectory timestamps exactly match input timestamps and never exceed a
  query timestamp.
- Presence intervals match emitted lifecycle transitions.
- Profile/component mismatches, unknown profiles, and illegal combinations fail
  closed.
- Temporal exceptions restore a deep state fingerprint rather than only a shallow
  `__dict__` snapshot.

### T1 exactness

The production T1 gate is stronger than the existing synthetic non-interference
test:

- Run frozen v1 in a separate process on frozen real inputs.
- Run A0-A4 and emit cumulative audit artifacts at every selected checkpoint.
- Compare complete file inventories and SHA-256 values for checkpoints, entities,
  meshes, and neutral exports.
- Hash the cumulative implementation's transitive source dependencies against a
  trusted baseline allowlist.
- Mutate every temporal configuration axis and verify that the temporal algorithm
  hash changes while the non-temporal hash and cumulative artifacts do not.
- Inject temporal failures and verify exact deep rollback.

Any cumulative mismatch rejects the candidate regardless of its T2 score.

### T2 mechanism gates

Before a full run, short-prefix and synthetic tests must show nonzero opportunities
and correct triggers for absence, readout invalidation, proposal recovery, motion
rejection, background reclaim, and eligible re-ID. If a mechanism has no opportunity
in TESSE-CD, it is reported as unavailable and is not used as a TESSE-CD claim.

Apartment development must reach at least 80 percent anchor coverage before a
candidate can run the complete official and common evaluation stack.

### Metric promotion gates

Apartment is the only development scene. Per-scene comparison values are resolved
from the frozen canonical baseline result packages before tuning; two-scene macro
values are not used as Apartment thresholds.

A candidate advances only when all of the following hold:

- Object F1 and current-map mIoU do not fall below the frozen OVIV2 release values.
- Dynamic F1 and Change F1 each exceed the strongest Apartment non-oracle baseline.
- Ghost is lower, BG F5 is higher, and recovery is shorter than the strongest
  Apartment non-oracle result under the same common-v2 scope.
- Every mechanism increment improves its primary metric without breaking an earlier
  gate.
- No weighted aggregate compensates for a failed metric.

The selected candidate must also satisfy these frozen OVIV2 T4 upper bounds:

- Total runtime no greater than 6.42 seconds per frame.
- Query mean no greater than 11.92 ms.
- Query p95 no greater than 12.12 ms.
- GPU memory no greater than 12.76 GB.
- RAM no greater than 9.36 GB.
- Map size no greater than 46.77 MB.

## Experiment Protocol

1. Freeze evaluator, schedule, baseline result, seed, tolerance, source, and artifact
   hashes.
2. Correct and validate the per-frame causal exporter on A0.
3. Run synthetic and short-prefix A0-A4 mechanism tests.
4. Run Apartment A0-A4 once for deterministic screening.
5. Run only non-dominated candidates through the full Apartment evaluation stack.
6. For randomized components, use five pre-registered fixed seeds. For deterministic
   components, run once and replay the evaluator twice byte-identically; do not
   manufacture seed confidence intervals.
7. Run complete T1 and T4 gates on the selected candidate.
8. Freeze code, configuration, seed list, evaluator, and artifact hashes.
9. Run Office once with the frozen package. Infrastructure failures may be retried
   only with identical hashes and without observing partial metrics.
10. Report Office failures and negative results without retuning.

Paired statistics use identical scenes, events, frames, checkpoints, and seeds.
Official F1 uncertainty is clustered by GT entity or trajectory. Common-v2
uncertainty is clustered by scene and event. Censored background and recovery events
are listed explicitly and are not treated as ordinary successful observations.

## Parallel Execution

The three GPUs are used only after short-prefix gates pass:

- GPU 0 runs A0/A1 or exporter-control candidates.
- GPU 1 runs A2 proposal and object-geometry candidates.
- GPU 2 runs A3/A4 background and re-ID candidates.

Each candidate has an independent output root and immutable config. Parallelism is
also constrained by available RAM and CPU because cached frontends make much of the
temporal run CPU-bound. Office remains locked until one candidate passes all
Apartment, T1, and T4 gates.

## Claim-Evidence Contract

| Claim | Required evidence |
| --- | --- |
| Correct causal dynamic output | Per-frame trajectory audit, explicit method-side dynamic state, frozen official evaluator |
| Better current-state maintenance | Dynamic, Change, Ghost, BG F5, and recovery all pass their independent gates |
| Occlusion safety | Occlusion retention and false-removal diagnostics with nonzero eligible anchors |
| Stable open-vocabulary quality | Object F1, current mIoU, and complete T1 exactness gate |
| Reversible background | A3 versus masking-only micro-ablation plus block rebuild tests |
| Identity retention | Eligible reactivation and ID-switch evidence; otherwise unavailable |
| Generalization | Frozen one-shot Office result |
| Practicality | Frozen T4 upper bounds and resource provenance |

No paper claim may exceed the evidence available under this contract.
