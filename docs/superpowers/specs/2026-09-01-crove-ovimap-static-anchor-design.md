# CROVE OVI-MAP Static Anchor Design

## Objective

Use an OVI-MAP map built only from the causal pre-intervention prefix as an
immutable static substrate for CROVE's TESSE-CD current-state readout. Preserve
CROVE's temporal association, lifecycle, re-identification, geometry epochs,
and reversible background evidence while preventing temporal fragmentation from
rewriting stable static instance geometry or semantics.

## Scope

The first proof run is TESSE-CD Apartment. Its first intervention is frame 263,
so the anchor may consume frames 0 through 262 and no later frame. Office is
eligible only after Apartment passes all quality gates. Replica/T1 execution,
the cumulative runtime, and every file bound by
`configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json` remain
byte-identical.

## Architecture

The method has three independent authorities:

1. **OVI-MAP anchor:** immutable instance geometry, background geometry,
   semantic feature prototypes, and semantic labels produced from the causal
   prefix.
2. **CROVE temporal state:** current entities, lifecycle evidence, motion,
   re-identification, geometry epochs, and reversible background changes.
3. **Anchor overlay state:** persistent temporal-to-anchor bindings and causal
   suppression decisions. This state never edits the anchor.

At checkpoint time, the current map is

`visible anchor entities - causally suppressed anchor entities + current temporal entities`.

An anchor entity remains authoritative while a bound temporal entity is
stationary or merely occluded. A moved entity suppresses its old anchor geometry
only after CROVE's existing motion and lifecycle evidence confirms the change;
its current temporal geometry is then emitted. A removed entity suppresses the
anchor only after confirmed visible absence. A new temporal entity is emitted
without modifying the anchor. Absence in one frame cannot suppress an anchor.

## Artifact Contract

The anchor package contains:

- a neutral `MapSnapshot` NPZ and JSONL pair;
- a JSON manifest binding the source RGB-D export, causal schedule, exact frame
  interval, OVI-MAP source commit, mapper outputs, color log, and all output
  hashes;
- `causal_cutoff_frame=262` for Apartment;
- `max_source_frame=262`, verified from the complete consumed-frame inventory;
- an immutable anchor identifier derived from canonical manifest content.

Loading fails closed on a path, hash, byte-count, scene, timestamp, frame-range,
or inventory mismatch. Pickle is accepted only by the one-time OVI-MAP adapter;
the runtime consumes the repository-owned NPZ/JSONL package with
`allow_pickle=False`.

## Association And Suppression

Initial bindings use deterministic one-to-one assignment. Candidate pairs must
pass spatial proximity and semantic compatibility gates. Their score is a
weighted sum of voxelized 3D IoU, centroid similarity, and cosine semantic
similarity. Assignment ordering and ties are deterministic.

Bindings are sticky across checkpoints and follow CROVE temporal identity, not
geometry epoch. Rebinding is allowed only when the old temporal identity is
expired and the replacement passes a stricter re-identification gate. Anchor
suppression requires a bound identity plus an existing CROVE-confirmed moved or
removed state. The overlay does not infer deletion solely from missing
detections or semantic disagreement.

## Readout Rules

- `UNCHANGED` or `OCCLUDED`: emit the anchor entity; suppress its temporal
  duplicate.
- `MOVED`: suppress the old anchor entity; emit the temporal entity at its
  current epoch.
- `REMOVED`: suppress the anchor entity; emit no replacement.
- `NEW`: emit the temporal entity in addition to all unsuppressed anchors.
- Unbound temporal entities are emitted with their CROVE identity.
- OVI-MAP background remains the static background. CROVE's committed current
  background additions are merged only through deterministic voxel
  deduplication; staged or uncertain ledger entries are excluded.

Every output entity records its authority (`ovimap_anchor` or
`crove_temporal`), anchor ID when applicable, temporal ID when applicable,
overlay state, and anchor manifest hash.

## Failure Handling

The run stops before publication if the anchor is missing, non-causal, mutated,
for another scene, or inconsistent with the requested checkpoint. Composition
is transactional: validation completes before any output is written, and
publication uses the existing atomic neutral snapshot exporter. The original
CROVE checkpoint remains available for audit.

## Evaluation Gates

Apartment is compared against the source-bound A6 baseline:

- Obj. F1 must be at least `0.372762`;
- current mIoU must be at least `0.142897`;
- Chg. F1 must exceed `0.060853`;
- ghost rate must be below `0.646883`;
- Dyn. F1 must be finite;
- all 1,745 frames and all official/common-v2 checkpoints must be present;
- the T1 protected-source verifier and T1 noninterference tests must pass.

Failure of any hard gate retains A6 and blocks the Office run. No failed or
estimated value may update paper tables.

## Publication Positioning

The prototype is reported as a composed method until the OVI-MAP substrate is a
native part of CROVE. The defensible claim is causal static-anchor/dynamic-delta
mapping, not a new OVI-MAP implementation. Novelty claims require a separate
prior-art search before manuscript use.
