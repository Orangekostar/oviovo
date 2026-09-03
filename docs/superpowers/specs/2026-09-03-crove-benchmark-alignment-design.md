# CROVE Benchmark Alignment Audit Design

Date: 2026-09-03

Status: approved by the user-provided master prompt before implementation

Master prompt SHA-256: `c878fe1d09e603a9bc8171afb048dd86a1fb8dbe371bcb6513a589d30a2d1c7c`

Base branch: `research/crove-localized-current-ownership`

Base SHA: `668aefc49034ef97d090b811b1c3f291ecafbd66`

Working branch: `research/crove-benchmark-alignment-audit`

## Objective

Determine, without benchmark shopping, whether CROVE's TESSE-CD limitation is
caused by the mapper, the open-vocabulary frontend, unequal input conditions,
the local Khronos adapter/aggregation, a mismatch between a historical 4D task
and the paper's causal current-state claim, or a combination of these factors.
The study may change the paper's benchmark hierarchy only from protocol-frozen
evidence collected in stages B0 through B9.

## Non-Negotiable Boundaries

- Do not alter A6, P5, P6-C, or P6-D historical artifacts or their frozen gates.
- Do not use future observations, evaluator output, GT object transforms, GT
  identities, GT semantics, or GT change labels inside a ranking-eligible
  method run.
- Label every GT semantic/instance condition `ORACLE_DIAGNOSTIC` and
  `NOT_RANKING_ELIGIBLE`.
- Keep official/post-release Khronos metrics and every diagnostic temporal
  slice under different protocol IDs and output roots.
- Never copy a paper number into a locally reproduced row.
- Freeze source commits, configs, inputs, selection rules, tolerances, and
  artifact hashes before viewing CROVE pilot results.
- Store only code, tests, manifests, patches, small summaries, and reports in
  Git. Keep RGB-D, meshes, checkpoints, maps, and large result trees external.
- Preserve the frozen T1 source gate and the 5,748-pass baseline.

## Existing Evidence Boundary

The frozen Apartment evidence supports only these conclusions:

| Evidence | Measured result | Allowed conclusion |
| --- | --- | --- |
| P6-A | 0/24 variants pass; Dyn is invariant at 0.069225 | P5 Object and Ghost effects arise from different suppression groups |
| P6-C | Obj 0.367307, Dyn 0.069225, Chg 0.087246, current 0.149449, Ghost 0.441689 | localized 5 cm ownership does not recover the frozen Object floor |
| P6-D | 0/104 dense proposals accepted | rigid centroid-translated anchor geometry is unsupported by the frozen agreement gate |
| P6-E | exact 0%, ambiguous 100% of 42,175 Dyn mass | final CSVs cannot identify a dominant Dyn mechanism |

Office remains `NOT_RUN_HELD_OUT`. These results do not prove that TESSE-CD is
invalid, that 3RScan or Flat is better, or that identity redesign is required.

## Claim-Evidence Matrix

| Claim under audit | Reviewer question | Required evidence | Primary metrics | Failure interpretation |
| --- | --- | --- | --- | --- |
| Causal current-state semantics | Does the map describe `M_T(T)` without future input? | timestamp-domain audit plus TESSE current diagonal | current Object/Dyn/Change F1, current mIoU, Ghost | protocol or method failure, classified from exact provenance |
| Fair temporal mapping | Is CROVE compared against the same perception condition? | Khronos GT/OpenSet and CROVE open/GT diagnostic matrix | Obj, Dyn, Change F1 and gap closure | frontend- versus mapper/protocol-dominated |
| Official metric validity | Does local aggregation reproduce its frozen upstream implementation? | source-bound row/grid and numeric parity on synthetic and real CSVs | row keys, slice values, F1, weighting | adapter/aggregator mismatch blocks method conclusions |
| Long-term identity/change | Can identity and moved/added/removed errors be located exactly? | causal 3RScan pilot with fixed IDs | t-AP, t-REC, ID switches, false Re-ID, change recall | dataset access or identity/mapper limitation |
| Controlled current maintenance | Are signed visibility and ownership mechanisms directly sensitive? | Flat run1-to-run2 causal pilot | current object P/R, stale FP, geometry F@5cm, recovery | mechanism weakness if Flat is valid and no gain appears |

Replica and ScanNet200 remain static-capability calibration only.

## Stage Order and Hard Stops

Execution order is strictly B0, B1, B2, B3, B4, B5, B6, B7, B8, B9.

- B0 freezes repository, claim, evaluator, data, and prior-artifact identities.
- B1 freezes the literature audit and pre-result suitability scorecard.
- B2 attempts a Khronos paper-like reproduction.
- B3 compares local and upstream aggregation.
- B4 separates frontend and temporal-map gaps.
- B5 adds evaluator-native exact attribution without changing official CSVs.
- B6 compares official/post-release output, the explicit current diagonal, and
  common-v2 current-state metrics on frozen variants.
- B7 runs the deterministic 3RScan pilot when its required assets exist.
- B8 runs the Flat pilot when its required assets exist.
- B9 applies only the frozen decision tree.

If B2 or B3 fails, benchmark validity is repaired or explicitly blocked before
any method-strength conclusion. Missing licensed assets produce
`BLOCKED_DATASET_ACCESS` reports and synthetic adapter tests; they do not permit
invented pilot metrics.

## Source and Protocol Identities

### Khronos

Record three identities independently:

1. RSS 2024 public release commit
   `742227a88de8b2ac23ac54d719b321c3af88dc75`.
2. Latest fetched official `main` commit at execution time.
3. The code and binary identity actually used by each current TESSE artifact.

An artifact is not assigned identity 3 merely because a workspace currently
points at a commit. The evidence must bind the evaluator executable and all
source/config inputs that affect its output. Otherwise report
`UNPROVEN_ARTIFACT_EVALUATOR_IDENTITY` and identify the strongest recoverable
base-plus-patch description.

Khronos reproduction status is exactly one of:

- `PAPER_PROTOCOL_EXACT`
- `PAPER_PROTOCOL_APPROX_PUBLIC`
- `BLOCKED_PAPER_ASSET`
- `BENCHMARK_REPRODUCTION_NO_GO`

The paper-like target is Apartment, GT pose, simulator GT semantics, 8 cm
resolution, and 5 m sensing range. Before the first complete run, the absolute
F1 tolerance is frozen at `0.02` for Background 0.912, Object 0.753, Dynamic
0.841, and Change 0.646. A source or data mismatch prevents an exact label even
when values fall within tolerance.

### Khronos Aggregation Parity

The parity layer consumes raw static, dynamic, and background CSVs and emits:

```python
@dataclass(frozen=True)
class KhronosParityResult:
    row_grid_equal: bool
    duplicate_policy_equal: bool
    slice_values_equal: bool
    metric_values_equal: bool
    local_metrics: dict[str, float]
    upstream_metrics: dict[str, float]
    max_abs_delta: float
```

It evaluates synthetic hand-checked rows, any public reference artifact, and
frozen A6/P5 rows. It must compare `(Name, Query)` keys, duplicates, robot/query/
online/full slices, per-state F1, background expansion/weighting, and aggregate
values. Numeric parity uses `rtol=0` and `atol=1e-12`; row and duplicate parity
are exact. A mismatch is `TESSE_ADAPTER_OR_AGGREGATOR_MISMATCH` and cannot be
hidden by changing old summaries.

### Frontend Fairness

The frozen matrix is K0 Khronos GT semantics, K1 Khronos OpenSet when
reproducible, C0 CROVE/A6 open-vocabulary, and C1 CROVE GT semantics oracle.
C2 GT instances and S0 shared observations are optional diagnostics only.

For metric `m` where larger is better:

```text
gap_open(m) = Khronos_GT(m) - CROVE_open(m)
oracle_gain(m) = CROVE_GT(m) - CROVE_open(m)
closure(m) = oracle_gain(m) / gap_open(m)
```

Only positive finite `gap_open` values participate. `FRONTEND_DOMINATED` means
the oracle closes at least 50% of the absolute open gap on the majority of
available headline metrics. Otherwise the result is
`MAPPER_OR_PROTOCOL_DOMINATED`; insufficient reproducible cells are
`INCONCLUSIVE_MISSING_CONDITION`.

### Evaluator-Native Attribution

The Khronos patch writes JSONL sidecars containing map name, robot/query time,
trajectory timestamp, predicted node ID, GT node ID, distance, metric type,
and TP/FP/FN status. Its official CSV bytes must be identical between patched
and unpatched runs on the same frozen input. The patch is rejected if any
official output differs. The repository stores only the patch and source
manifest, not a third-party source tree.

### TESSE Current Diagonal

`TESSE_CURRENT_DIAGONAL` is a diagnostic protocol. It uses the same frozen GT,
predictions, association thresholds, and semantic conditions as its paired
Khronos run, but selects only `belief_time == robot_time`. It records both time
domains explicitly and never infers diagonal membership from an unverified
column name. It outputs current Object, Dynamic, and Change F1 plus exact
association sidecars. Existing current mIoU, Ghost, Background F@5cm, and
Recovery frames remain separate common-v2 diagnostics.

## Frozen Full-vs-Current Analysis

The only variants are A6, c553 static-anchor, P5, and P6-C. No mapper tuning or
new training is allowed. A material mismatch requires at least one of:

1. a systematic direction reversal between official and current headline
   metrics for A6 versus P5/P6;
2. stable gains on multiple current metrics while official historical metrics
   remain unresponsive or reverse;
3. direct source/row-mass evidence that the full score is dominated by
   retrospective states outside the paper's bounded claim.

One favorable cell is insufficient.

## 3RScan Pilot

### Input and Selection

The local metadata source is `/home/ww/vv/dataset/3RScan/3RScan.json`, observed
SHA-256 `674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64`.
Selection is frozen before method results:

1. use environments whose metadata `type` is `validation`;
2. require every selected visit ID to appear in the frozen validation scan
   list;
3. require at least one rescan and at least one registered rigid, nonrigid, or
   removed change across rescans;
4. sort by reference scan UUID;
5. take the first 10 environments.

The selection manifest binds metadata, validation list, selected visits, and
the rule. A missing required RGB-D/pose/mesh/annotation member blocks only the
runtime portion whose contract needs that member.

### Causal and Metric Contract

At session `s`, the mapper may consume only sessions `0..s`; GT global
alignment is allowed only under the pose condition shared by every compared
method. GT object transforms and cross-time IDs are evaluator-only.

The adapter returns immutable `RScanEnvironment`, `RScanSession`, and
`RScanChange` records. The evaluator separates community-compatible stage AP,
t-AP, and t-REC from custom exact-ID diagnostics: ID switches, false Re-ID,
reactivation recall, current-object recall, stale-object FP, and recall by
rigid/nonrigid/added/removed type. Open-vocabulary label mapping to RIO classes
uses a separately hash-bound deterministic crosswalk.

## Panoptic Mapping Flat Pilot

The adapter validates two ordered trajectories and color, depth, panoptic GT,
predicted panoptic, label, pose, timestamp, change-log, and structural GT
members. Run 1 is fully processed before run 2; future run-2 data cannot affect
earlier checkpoints.

Conditions are separate:

- `Flat-GT-Panoptic`: `ORACLE_DIAGNOSTIC`, mechanism only.
- `Flat-Predicted-Panoptic`: non-oracle perception condition.
- CROVE open-vocabulary frontend: optional third condition, never merged into
  the first two ranking columns.

Fixed variants are A6, P5, and P6-C. Metrics are current object precision/
recall, moved/added/removed change, stale geometry FP, current geometry F@5cm,
background/free-space recovery, and recovery latency. If public Flat assets
are absent or access-controlled, the report is `BLOCKED_DATASET_ACCESS` after
synthetic adapter/evaluator tests pass.

## Pre-Result Suitability Scorecard

Before any CROVE 3RScan or Flat score is viewed, commit one JSON document with
integer scores 0, 1, or 2 for TESSE official, TESSE current diagonal, 3RScan,
and Panoptic Flat across exactly these dimensions:

```text
matches_current_state_claim
online_causal_compatibility
open_vocab_input_fairness
short_term_dynamics_coverage
long_term_change_coverage
exact_cross_time_identity_gt
change_type_gt
geometry_current_surface_gt
evaluator_error_provenance
official_code_reproducibility
baseline_ecosystem
real_world_relevance
```

Every score carries at least one source ID resolved in the literature report.
The JSON includes its own content hash in the decision ledger rather than a
self-referential field.

## Decision Tree

B9 may emit only one of:

- `KEEP_TESSE_HEADLINE`
- `KEEP_TESSE_HEADLINE_WITH_SPLIT_PROTOCOL`
- `DEMOTE_TESSE_TO_STRESS_TEST`
- `REPLACE_HEADLINE_DYNAMIC_SUITE`
- `BENCHMARK_OK_METHOD_WEAK`

The choice follows the exact conditions in the master prompt. A Pareto report
may guide future research selection but cannot modify the historical
all-or-nothing gate or hide any reported metric.

## Reports and Decision Ledger

Every B stage appends a ledger entry containing Question, Evidence before,
Frozen source, Frozen protocol, Command, Result, Deviation from paper,
Decision, What this rules in/out, Commit, and Artifacts. Failed or blocked
stages remain in the ledger.

Required final reports are the Khronos protocol reproduction, Khronos metric
parity, TESSE input fairness, TESSE current-vs-full4d, 3RScan pilot, Panoptic
Flat pilot, literature benchmark audit, benchmark decision, and final handoff.
The handoff answers all 15 questions in the master prompt and binds external
large artifacts by path, SHA-256, byte count, generation command, source
commit, and config hash.

## Verification

Each new pure function is introduced through a failing test. Adapters use
synthetic fixtures for schema, ordering, tamper, causality, and missing-asset
behavior. Khronos tests include upstream-compatible fixtures and CSV
byte-identity checks for instrumentation. Focused historical suites, the T1
protected-source verifier, `git diff --check`, `python -m compileall -q src
scripts tests`, and full `pytest -q` are mandatory before upload.

The final branch is complete only after every required report exists, all
available stages have evidence-backed terminal states, every missing external
asset is explicitly blocked rather than guessed, all expected files are
committed, and local `HEAD` equals the pushed GitHub branch SHA.
