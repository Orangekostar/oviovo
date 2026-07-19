# Benchmark Repair, OVIV2 Validation, and ScanNet Preparation Design

## Purpose

This design resolves the credibility gaps found in Table 1 while continuing the current OVIV2 mainline and preparing the licensed ScanNet200 work without fabricating unavailable data.

The current repository state already contains the OVIV2 sparse voxel core, reversible ownership, atomic snapshots, labeled meshing, Replica evaluation, Replica-8 execution, and a verified result at commit `eed1722`. This design therefore does not repeat those completed tasks. It audits their metric contract against the released OVI-MAP evaluator, repairs the external baseline adapters, and defines the next dynamic-mainline boundary.

## Goals

1. Repair OVI-MAP class-agnostic instance evaluation so that every reconstructed global instance can participate independently of semantic-feature availability.
2. Preserve a strict neutral evaluator for headline comparisons while also producing non-headline official-style diagnostics that explain differences from published numbers.
3. Replace the current ConceptGraphs streamlined run as the canonical headline candidate with a run using the released paper configuration.
4. Audit the current OVIV2 Replica evaluator and published result against the same explicit projection, filtering, and AP definitions.
5. Prepare ScanNet200 validation, conversion, runner, and result infrastructure without bypassing publisher access terms or freezing an unvalidated manifest.
6. Keep all changes isolated from the dirty legacy mapping and pipeline files already present in the worktree.

## Non-Goals

- Do not copy paper values into the registry.
- Do not use GT semantic, instance, visibility, or class-presence information at runtime.
- Do not use the ConceptGraphs evaluator's per-scene GT-only class suppression for headline results.
- Do not overwrite existing result directories or mutate reference checkouts.
- Do not download ScanNet data from an unlicensed mirror.
- Do not refactor `src/modules/`, `src/pipelines/`, or the existing dirty legacy implementation as part of these workstreams.
- Do not optimize runtime until the metric and artifact contracts are stable.

## Workstream Boundaries

### Workstream A: Baseline Credibility Repair

This workstream owns external baseline loading, neutral evaluation, official-style diagnostics, and corrected result manifests.

Allowed implementation paths:

- `src/evaluation/baselines/`
- `scripts/evaluation/`
- `tests/evaluation/`
- `docs/paper/results/baselines/`
- external wrappers and outputs under `/home/ww/oviovo_baseline_runs`

It must not modify the OVI-MAP or ConceptGraphs reference checkouts. Compatibility changes remain in external build copies.

### Workstream B: OVIV2 Mainline Validation

This workstream owns the metric audit of the completed voxel-first static pipeline and the transition to dynamic current-state evaluation.

Allowed implementation paths:

- `src/oviv2/`
- `src/evaluation/oviv2_replica.py`
- `src/evaluation/oviv2_result.py`
- `scripts/evaluation/evaluate_oviv2_replica.py`
- `tests/oviv2/`
- focused OVIV2 evaluation tests and result manifests

Existing OVIV2 results remain immutable. Any corrected contract produces a new run ID and result directory.

### Workstream C: ScanNet200 Preparation

This workstream owns data validation and execution preparation only. It cannot produce a frozen test manifest or verified metric until publisher-approved data is present.

Allowed implementation paths:

- `src/evaluation/datasets/` or the nearest existing evaluation-data package
- `scripts/evaluation/`
- `tests/evaluation/`
- `configs/evaluation/manifests/` after real data validation
- `/home/ww/oviovo_benchmark_assets/scannet200`
- `/home/ww/oviovo_baseline_runs/.../scannet200`

## OVI-MAP Data Model

OVI-MAP exposes two distinct populations that must not be conflated:

1. The global instance mesh and `LogInstanceColor` records define reconstructed class-agnostic instances.
2. The `inst_sem_*.pkl` file defines the subset with selected semantic views and embeddings.

The neutral adapter will construct every mesh-backed global instance as an entity. If an entity has no semantic feature record, its semantic label and embedding remain absent, but its geometry still participates in class-agnostic instance evaluation.

Semantic evaluation retains the released OVI-MAP rule requiring at least two accepted views. This gate affects only semantic labeling and semantic-class-constrained diagnostics. It must not remove an entity from class-agnostic AP.

Color identity comes from the authoritative `Instance: <id> Color: (<r>,<g>,<b>)` backend log. Raw input mask colors are not treated as final mesh colors.

## OVI-MAP Metric Outputs

Each corrected scene produces two explicitly separated reports:

- `neutral_metrics.json`: strict frozen-vocabulary evaluation used by the benchmark registry.
- `official_style_diagnostics.json`: released OVI-MAP projection and threshold statistics used only to explain reproduction differences.

The neutral report keeps the shared 5 cm projection threshold and auditable unmatched domain. The diagnostic report records the released evaluator's full output names rather than relabeling precision or recall as AP.

Every report includes:

- mesh-backed instance count;
- semantic-feature instance count;
- instances with at least two accepted views;
- predicted and GT vertex counts;
- matched vertex ratio;
- vocabulary and excluded-class lists;
- projection threshold and strict/inclusive comparison rule;
- per-class and macro metrics;
- source artifact paths and SHA-256 values.

## ConceptGraphs Canonical Run

The existing YOLO-World plus MobileSAM result is retained as a streamlined diagnostic. It cannot silently represent the paper's canonical ConceptGraphs configuration.

The canonical run uses the released configuration:

- SAM segment-all frontend for the `ConceptGraphs` row;
- released Grounded-SAM dependency commit and required weights;
- released object-mapping thresholds and merge settings;
- the frozen benchmark frame selection for the neutral headline run;
- a separate stride-5 official-style diagnostic when needed for paper comparison.

The canonical run first executes room0 and must pass non-empty detection, object-map, geometry, and semantic-coverage gates. Only then may it expand to the remaining seven scenes.

The official `n_exclude=6` and GT-existing-class behavior is diagnostic only. The neutral result continues to evaluate the frozen vocabulary without test-GT class suppression.

## OVIV2 Evaluator Audit

The current evaluator is treated as implemented but not exempt from protocol audit. The audit compares it against both the frozen benchmark requirements and the released OVI-MAP behavior.

The following invariants must be recorded and tested:

- projection uses nearest predicted mesh vertex per GT vertex;
- the headline distance rule is exactly specified and tested at the threshold boundary;
- unmatched vertices receive zero semantic and entity IDs;
- semantic metrics operate on the declared valid class domain;
- instance AP is computed per semantic class with deterministic ranking;
- minimum-region and accepted-view filters are explicit;
- geometry F5 is computed from raw predicted and GT geometry before semantic filtering;
- no evaluator-only GT data enters the mapper or entity feature pipeline.

If the audit changes a metric definition, the existing `20260719-s10-200f` result remains unchanged and a new versioned result is generated. Registry pointers move only after the new result passes provenance and hash verification.

After the static evaluator contract is stable, the mainline moves to dynamic current-state snapshots. Dynamic work begins with synthetic intervention fixtures and Replica-derived change sequences before TESSE-CD is available.

## ScanNet200 Preparation

The validator accepts a publisher-approved dataset root and verifies the five predeclared candidate scenes:

- `scene0011_00`
- `scene0050_00`
- `scene0231_00`
- `scene0378_00`
- `scene0518_00`

For each scene it verifies required RGB-D or preprocessed inputs, pose data, semantic labels, instance labels, mesh files, and deterministic hashes. It rejects symlink targets outside the declared dataset root unless explicitly allowed as a recorded data view.

The formal `scannet200_5.json` manifest is written only after every selected scene passes validation. The manifest freezes scene IDs, file hashes, frame selection, ScanNet200 vocabulary, aliases, depth scale, pose source, and split role.

Before data access, implementation may provide:

- validator tests using synthetic directory fixtures;
- conversion command generation;
- method-specific runner templates;
- weight/config preflight;
- structured blocker updates.

It may not provide metrics, empty placeholder manifests, or a status stronger than `BLOCKED`.

## Data and Result Flow

1. Raw upstream artifacts remain immutable.
2. Method-specific loaders produce neutral evaluation objects and audit metadata.
3. Neutral evaluators write per-scene metrics and diagnostics.
4. Aggregators macro-average only the frozen scene set.
5. Result finalizers hash every config, weight, manifest, log, raw output, and aggregate.
6. Importers read values only through result-manifest JSON pointers.
7. Existing registry values change only after a new result is independently verifiable.

Official-style diagnostics never bind headline tokens.

## Failure Handling

- Missing mesh colors, duplicate instance IDs, or mismatched color logs are hard errors.
- Missing semantic features do not invalidate class-agnostic geometry; they remain explicit unlabeled entities.
- A canonical ConceptGraphs room0 gate failure blocks the long run and preserves its command, exit code, environment, and logs.
- ScanNet access, license, or payload failures update a structured blocker and leave the registry `UNFILLED`.
- Metric-contract changes never rewrite an existing result directory.
- Reference-repository dirtiness stops finalization until explicitly resolved.

## Testing Strategy

All production behavior changes use test-first development.

Baseline tests cover:

- more mesh instances than semantic-feature records;
- featureless instances participating in class-agnostic AP;
- featureless instances excluded from semantic classification;
- duplicate or absent color-log mappings;
- exact neutral/diagnostic separation;
- official output parsing without metric renaming.

ConceptGraphs tests cover:

- canonical configuration identity;
- room0 gate validation;
- explicit separation of neutral and `n_exclude=6` diagnostics;
- rejection of GT-only class suppression in headline mode.

OVIV2 tests cover:

- threshold boundary projection;
- hand-computed semantic confusion;
- class-constrained AP fixtures;
- accepted-view and minimum-region filtering;
- mesh and snapshot immutability;
- result provenance and hash verification.

ScanNet tests cover:

- missing files and scenes;
- wrong hashes and invalid label domains;
- path escape through symlinks;
- manifest refusal before complete validation;
- deterministic manifest generation after a valid synthetic fixture.

## Acceptance Criteria

1. OVI-MAP room0 reports the full mesh-backed instance population separately from the semantic-feature subset.
2. OVI-MAP neutral instance metrics no longer depend on semantic-feature availability.
3. Official OVI-MAP evaluator output is captured as diagnostic evidence without populating headline tokens.
4. ConceptGraphs canonical room0 passes all gates before the eight-scene run starts.
5. The current OVIV2 evaluator contract has explicit passing boundary and hand-computed tests.
6. Any changed OVIV2 headline result uses a new run ID and complete provenance.
7. ScanNet infrastructure is testable without data, while formal manifest creation remains blocked until licensed data validation.
8. Canonical and derived table generation tests pass, OVIOVO/OVIV2 token ownership remains enforced, and every populated cell resolves through its JSON pointer.
9. `VERIFY_ENVIRONMENTS=0 bash scripts/verify_workspace.sh` passes with all 19 reference repositories clean.
10. No pre-existing dirty legacy mapping or pipeline file is staged or overwritten.

## Execution Order

The program is decomposed into three implementation plans:

1. OVI-MAP adapter repair and ConceptGraphs canonical gate.
2. OVIV2 evaluator audit and dynamic snapshot transition.
3. ScanNet200 pre-data validator and runner preparation.

Inline execution uses review checkpoints between plans. The first implementation target is the OVI-MAP adapter because it contains a confirmed result-validity defect and can be re-evaluated from existing artifacts without another mapping run.
