# OVIV2 View-Consensus Pareto Instance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GT-free multi-view instance proposal pyramid that raises OVIV2 AP50 while preserving the exact v5 primary prediction prefix and preventing every per-scene AP25/AP50 regression.

**Architecture:** Reconstruct the formal v5 projected predictions from the frozen primary and auxiliary snapshots, deduplicate them exactly once into an immutable prefix, then append independently screened view-consensus candidates below the minimum primary score. New modules own ordered AP evaluation, sparse observation graph construction, proposal formation, visibility/mesh projection, provenance, and search; no v5-hashed source file is modified.

**Tech Stack:** Python 3.11, NumPy, SciPy `cKDTree`, existing OVIV2 `FrameObservation`/snapshot/mesh/Replica evaluator contracts, pytest, JSON/NPZ audit artifacts.

---

## File Map

- Create `src/evaluation/oviv2_ordered_instance.py`: primary-prefix composition, suffix-only deduplication, ordered AP evaluation, and prefix fingerprints.
- Create `src/evaluation/oviv2_view_graph.py`: immutable graph configuration/evidence and sparse inverted-voxel edge generation.
- Create `src/evaluation/oviv2_view_proposals.py`: consensus/core/union/hierarchical proposals, visibility pruning, mesh projection, and 2D+3D unions.
- Create `src/evaluation/oviv2_view_consensus_io.py`: frozen-v5 reconstruction, cache preflight, provenance hashing, and fresh atomic outputs.
- Create `scripts/evaluation/evaluate_oviv2_view_consensus.py`: one-scene orchestration and V0/V1-V5 evaluation.
- Create `scripts/evaluation/search_oviv2_view_consensus_room0.py`: staged room0 search and lexicographic freezing.
- Create `scripts/evaluation/promote_oviv2_view_consensus.py`: held-out per-scene and aggregate Pareto gate.
- Create `configs/evaluation/oviv2_view_consensus_replica8.json`: immutable artifact roots, v5 baselines, and bounded search grid.
- Create focused tests under `tests/evaluation/`; do not alter existing v5 tests or evaluators.

### Task 1: Ordered Primary-Prefix Evaluation

**Files:**
- Create: `src/evaluation/oviv2_ordered_instance.py`
- Create: `tests/evaluation/test_oviv2_ordered_instance.py`

- [ ] **Step 1: Write failing tests for immutable-prefix composition**

Use `ProjectedInstanceHypothesis` fixtures with six-vertex masks. Require primary support filtering plus `0.7` deduplication, suffix screening at `0.9`, strict suffix score separation, exact prefix identity, and rejection of unequal mask lengths.

```python
composition = compose_ordered_predictions(
    primary=(primary_a, duplicate_a, primary_b),
    suffix=(novel_c, duplicate_b),
    min_instance_vertices=2,
    primary_deduplication_iou=0.7,
    suffix_deduplication_iou=0.9,
)
assert composition.primary_ids == ("p:a", "p:b")
assert composition.ordered_ids[:2] == composition.primary_ids
assert composition.ordered_ids[2:] == ("vc:c",)
assert max(item.score for item in composition.suffix) < min(
    item.score for item in composition.primary
)
```

- [ ] **Step 2: Run RED**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_ordered_instance.py
```

Expected: collection fails with `ModuleNotFoundError: src.evaluation.oviv2_ordered_instance`.

- [ ] **Step 3: Implement immutable composition and fingerprints**

Define these public contracts exactly:

```python
@dataclass(frozen=True)
class OrderedPredictionComposition:
    primary: tuple[ProjectedInstanceHypothesis, ...]
    suffix: tuple[ProjectedInstanceHypothesis, ...]
    rejected_suffix_ids: tuple[str, ...]
    primary_sha256: str

    @property
    def ordered(self) -> tuple[ProjectedInstanceHypothesis, ...]:
        return self.primary + self.suffix

    @property
    def primary_ids(self) -> tuple[str, ...]:
        return tuple(item.hypothesis_id for item in self.primary)

    @property
    def ordered_ids(self) -> tuple[str, ...]:
        return tuple(item.hypothesis_id for item in self.ordered)


```

Add public functions `projected_iou(left, right) -> float`, `fingerprint_predictions(predictions) -> str`, and `compose_ordered_predictions(primary, suffix, *, min_instance_vertices, primary_deduplication_iou, suffix_deduplication_iou) -> OrderedPredictionComposition` with the argument types used by `OrderedPredictionComposition` above.

Implementation order is: validate equal mask length; support-filter both strata; call the existing `deduplicate_projected_hypotheses` only on primary; sort suffix by `(-score, hypothesis_id)`; reject a suffix candidate with IoU at or above the suffix threshold against any primary or earlier kept suffix; scale kept suffix scores by `nextafter(min_primary_score, 0.0) / max_suffix_score`; fingerprint IDs, IEEE-754 scores, and packed masks in order. Raise if primary is empty, its minimum score is not positive, or scaling cannot produce a strict separation.

- [ ] **Step 4: Add ordered AP parity and exhaustive nondecrease tests**

Implement `evaluate_ordered_projected_hypotheses(ordered: Sequence[ProjectedInstanceHypothesis], ground_truth: ReplicaGroundTruth, *, instance_semantic_ids: set[int], min_instance_vertices: int) -> dict[str, object]`.

Copy the formal GT-mask, IoU-matrix, greedy matching, `_average_precision`, recall, count, ID, and confidence logic into the new module, but do not sort or deduplicate `ordered`. Add a V0 parity test against `evaluate_projected_instance_hypotheses` and enumerate every binary TP/FP prefix and suffix of lengths zero through eight for GT counts one through four:

```python
for gt_count in range(1, 5):
    for prefix_bits in binary_sequences(max_length=8):
        baseline = ap_from_bits(prefix_bits, gt_count)
        for suffix_bits in binary_sequences(max_length=8):
            assert ap_from_bits(prefix_bits + suffix_bits, gt_count) + 1e-15 >= baseline
```

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/evaluation/test_oviv2_ordered_instance.py
git add src/evaluation/oviv2_ordered_instance.py tests/evaluation/test_oviv2_ordered_instance.py
git commit -m "feat: add Pareto-safe ordered instance evaluation"
```

Expected: all ordered-instance tests pass.

### Task 2: Sparse Multi-View Observation Graph

**Files:**
- Create: `src/evaluation/oviv2_view_graph.py`
- Create: `tests/evaluation/test_oviv2_view_graph.py`

- [ ] **Step 1: Write failing graph tests**

Create `FrameObservation` fixtures covering shared voxels, disjoint masks, same-frame masks, semantic disagreement, matching and mismatching feature-model IDs, and shuffled input order. Assert that only cross-frame compatible pairs are returned and output is deterministic.

```python
evidence = build_sparse_observation_edges(
    tuple(reversed(observations)),
    ViewGraphConfig(
        minimum_overlap_voxels=2,
    ),
)
edges = select_observation_edges(
    evidence,
    minimum_voxel_iou=0.05,
    minimum_directed_coverage=0.25,
    minimum_feature_cosine=0.20,
)
assert [(edge.left_id, edge.right_id) for edge in edges] == [(10, 21)]
assert edges[0].shared_voxels == 2
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_oviv2_view_graph.py
```

Expected: module import failure.

- [ ] **Step 3: Implement validated graph records and sparse pair generation**

Define:

```python
@dataclass(frozen=True)
class ViewGraphConfig:
    minimum_overlap_voxels: int = 4
    require_semantic_agreement: bool = True


@dataclass(frozen=True)
class ObservationEdge:
    left_id: int
    right_id: int
    shared_voxels: int
    voxel_iou: float
    left_coverage: float
    right_coverage: float
    feature_cosine: float | None
    view_direction_cosine: float | None


```

Add `build_sparse_observation_edges(observations: Sequence[FrameObservation], config: ViewGraphConfig) -> tuple[ObservationEdge, ...]` and `select_observation_edges(evidence: Sequence[ObservationEdge], *, minimum_voxel_iou: float, minimum_directed_coverage: float, minimum_feature_cosine: float) -> tuple[ObservationEdge, ...]`.

Filter to `ObservationKind.OBJECT` with positive semantic IDs. Build `dict[VoxelKey, list[int]]`, count candidate-pair overlap once per shared voxel, and never allocate an observation-by-observation dense matrix. Reject same-frame pairs and semantic mismatches before feature work. Use feature cosine only when both image features exist and `feature_model_id` matches; otherwise store `None`. Preserve every pair meeting `minimum_overlap_voxels` so later strong and weak thresholds reuse the same evidence. `select_observation_edges` admits a pair when either voxel IoU or both directed coverages pass and any available compatible feature passes; missing compatible features do not veto geometry. Sort both outputs by `(left_id, right_id)`.

- [ ] **Step 4: Add resource and validation tests**

Monkeypatch `np.zeros` to fail on a square shape equal to the observation count, then build a 1,000-node sparse fixture. Test non-finite thresholds, invalid ranges, duplicate observation IDs, and incompatible feature dimensions.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/evaluation/test_oviv2_view_graph.py
git add src/evaluation/oviv2_view_graph.py tests/evaluation/test_oviv2_view_graph.py
git commit -m "feat: build sparse OVIV2 multi-view observation graph"
```

Expected: all graph tests pass without a dense pair matrix.

### Task 3: Consensus Proposal Pyramid

**Files:**
- Create: `src/evaluation/oviv2_view_proposals.py`
- Create: `tests/evaluation/test_oviv2_view_proposals.py`

- [ ] **Step 1: Write failing component/core/union tests**

Construct three views of one object with a stable four-voxel core and one-view boundary voxels, plus a weak bridge to a second object. Require strong components, core votes, inclusive union, weak hierarchical merge, deterministic IDs, and proposal caps.

```python
proposals = build_proposal_pyramid(
    observations,
    evidence,
    ProposalPyramidConfig(
        strong_voxel_iou=0.05,
        strong_directed_coverage=0.25,
        minimum_feature_cosine=0.20,
        minimum_distinct_views=2,
        core_vote_fraction=2 / 3,
        inclusive_coverage=0.25,
        weak_merge_iou=0.10,
        minimum_voxels=3,
        maximum_proposals=64,
    ),
)
assert {item.kind for item in proposals} >= {"consensus", "core", "union"}
assert stable_core <= next(item.voxel_keys for item in proposals if item.kind == "core")
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_oviv2_view_proposals.py -k 'component or core or union'
```

Expected: module import failure.

- [ ] **Step 3: Implement proposal records and pyramid construction**

Define:

```python
ProposalKind = Literal["consensus", "core", "union", "hierarchical", "hybrid"]


@dataclass(frozen=True)
class ProposalPyramidConfig:
    strong_voxel_iou: float = 0.05
    strong_directed_coverage: float = 0.25
    minimum_feature_cosine: float = 0.20
    minimum_distinct_views: int = 2
    core_vote_fraction: float = 0.50
    inclusive_coverage: float = 0.25
    weak_merge_iou: float = 0.10
    minimum_voxels: int = 10
    maximum_proposals: int = 1024


@dataclass(frozen=True)
class VoxelProposal:
    proposal_id: str
    semantic_id: int
    kind: ProposalKind
    voxel_keys: frozenset[VoxelKey]
    observation_ids: tuple[int, ...]
    supporter_frame_ids: tuple[int, ...]
    mean_confidence: float
    consensus_density: float
    border_fraction: float


```

Add `build_proposal_pyramid(observations: Sequence[FrameObservation], evidence: Sequence[ObservationEdge], config: ProposalPyramidConfig) -> tuple[VoxelProposal, ...]`.

Select strong edges with `strong_voxel_iou`, `strong_directed_coverage`, and `minimum_feature_cosine`, then use deterministic union-find for strong connected components. Count voxel votes once per distinct frame, set the core threshold to `ceil(core_vote_fraction * view_count)`, form inclusive unions from observations whose directed coverage of the core reaches `inclusive_coverage`, and form hierarchical candidates only from same-semantic strong components connected by preserved evidence at `weak_merge_iou`. Deduplicate identical `(semantic_id, voxel_keys)` candidates by kind priority `core`, `consensus`, `union`, `hierarchical`, `hybrid`. Exceeding `maximum_proposals` raises instead of truncating.

- [ ] **Step 4: Add deterministic scoring tests**

Implement a GT-free raw score in `[0, 1]`:

```python
score = (
    0.30 * min(distinct_views / 8.0, 1.0)
    + 0.25 * mean_confidence
    + 0.25 * consensus_density
    + 0.20 * (1.0 - border_fraction)
)
```

Test shuffled inputs, exact tie ordering, empty graphs, single-view rejection, and all configuration range failures.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/evaluation/test_oviv2_view_proposals.py
git add src/evaluation/oviv2_view_proposals.py tests/evaluation/test_oviv2_view_proposals.py
git commit -m "feat: construct view-consensus proposal pyramid"
```

Expected: all proposal-pyramid tests pass.

### Task 4: Visibility Pruning, Mesh Projection, And 2D+3D Unions

**Files:**
- Modify: `src/evaluation/oviv2_view_proposals.py`
- Modify: `tests/evaluation/test_oviv2_view_proposals.py`

- [ ] **Step 1: Write failing visibility tests**

Use 4x4 synthetic frames with identity and translated poses. Test in-frustum projection, behind-camera rejection, depth occlusion, visible-point minimum, and supporter/observer ratio pruning.

```python
visibility = measure_proposal_visibility(
    points_xyz,
    frames,
    depth_tolerance_m=0.10,
    minimum_visible_points=2,
)
assert visibility.observer_frame_ids == (0, 2)
assert keep_by_visibility(
    supporter_frame_ids=(0, 2),
    visibility=visibility,
    minimum_supporter_ratio=0.67,
)
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_oviv2_view_proposals.py -k visibility
```

Expected: missing visibility API failures.

- [ ] **Step 3: Implement camera-depth visibility**

Define:

```python
@dataclass(frozen=True)
class ProposalVisibility:
    observer_frame_ids: tuple[int, ...]
    visible_point_counts: tuple[int, ...]


```

Add `measure_proposal_visibility(points_xyz: np.ndarray, frames: Sequence[Frame], *, depth_tolerance_m: float, minimum_visible_points: int) -> ProposalVisibility`.

Transform world points with `inv(frame.pose)`, project positive-z points using frame intrinsics, round to valid image pixels, and count a point visible when source depth is finite/positive and `camera_z <= source_depth + depth_tolerance_m`. A frame is an observer at the configured point count. The supporter ratio is `len(supporters intersect observers) / len(observers)`; zero observers reject the proposal.

- [ ] **Step 4: Write failing mesh projection and hybrid-union tests**

Use a labeled six-vertex mesh. Require exact voxel membership, one-cell Chebyshev dilation, sorted read-only indices, synthetic IDs starting at `1_000_000`, and union with a compatible primary hypothesis only when projected IoU is within `[0.05, 0.90)`.

```python
hypotheses = project_voxel_proposals_to_mesh(
    mesh,
    proposals,
    voxel_size_m=0.05,
    dilation_cells=1,
    synthetic_entity_id_start=1_000_000,
)
hybrids = build_hybrid_instance_hypotheses(
    hypotheses,
    primary_hypotheses,
    minimum_overlap_iou=0.05,
    maximum_overlap_iou=0.90,
)
assert all(item.hypothesis_id.startswith("vc:") for item in hypotheses + hybrids)
```

- [ ] **Step 5: Implement projection, pruning, and hybrid unions**

Map every mesh vertex to `floor(vertex_xyz / voxel_size_m)`, use an inverted voxel-to-vertex index, apply optional integer-cell dilation, and emit `InstanceHypothesis(kind="child")`. Reject candidates below the configured vertex minimum. Add visibility support and feature-agreement bonuses to the raw proposal score before suffix scaling; never read ground truth or evaluator output.

- [ ] **Step 6: Run GREEN and commit**

```bash
pytest -q tests/evaluation/test_oviv2_view_proposals.py
git add src/evaluation/oviv2_view_proposals.py tests/evaluation/test_oviv2_view_proposals.py
git commit -m "feat: project and prune view-consensus proposals"
```

Expected: all proposal and visibility tests pass.

### Task 5: Frozen-v5 Reconstruction And One-Scene CLI

**Files:**
- Create: `src/evaluation/oviv2_view_consensus_io.py`
- Create: `scripts/evaluation/evaluate_oviv2_view_consensus.py`
- Create: `tests/evaluation/test_oviv2_view_consensus_io.py`
- Create: `tests/evaluation/test_evaluate_oviv2_view_consensus_cli.py`

- [ ] **Step 1: Write failing v5 reconstruction tests**

Build tiny primary/auxiliary snapshots with the existing evaluator fixture. Require snapshot and run-manifest checksums, primary `InstanceHeadConfig(20, 8, 0.05, 0.7, 2, 1)`, auxiliary rank-after-primary behavior, and byte-stable prefix fingerprints.

```python
context = load_frozen_v5_context(
    primary_snapshot=primary_snapshot,
    auxiliary_snapshot=auxiliary_snapshot,
    auxiliary_run_manifest=auxiliary_manifest,
    semantic_evidence=semantic_evidence,
    semantic_replay_manifest=replay_manifest,
    benchmark_manifest=benchmark_manifest,
    scene="fixture",
)
assert context.instance_config.deduplication_iou_threshold == 0.7
assert context.snapshot_revision == 2
assert context.source_hashes.keys() >= {"ordered_instance", "view_graph", "view_proposals"}
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_oviv2_view_consensus_io.py tests/evaluation/test_evaluate_oviv2_view_consensus_cli.py
```

Expected: both new modules are missing.

- [ ] **Step 3: Implement immutable input and provenance contracts**

Define a frozen `FrozenV5Context` carrying primary mesh/hypotheses, auxiliary mesh/hypotheses, vocabulary, instance IDs, GT, dataset, frontend adapter, baseline metrics, and all input/source hashes. Reconstruct primary and auxiliary projected hypotheses with existing public OVIV2 functions. Require the generated V0 IDs, scores, AP25, and AP50 to equal the supplied v5 `metrics.json` within `1e-12`; otherwise raise `ValueError("V0 does not reproduce frozen v5")` before candidate generation.

The loader accepts these exact artifact families:

```text
primary: /home/ww/oviovo_final_outputs/oviv2_replica8_stage3_s049_200f_5f272a8/{scene}/final/oviv2_voxel_snapshot.npz
auxiliary: /home/ww/oviovo_experiments/20260722_route3_replica8_auxiliary/maps/{scene}/final/oviv2_voxel_snapshot.npz
frontend: /home/ww/oviovo_experiments/20260722_route3_replica8_auxiliary/frontend/{scene}/sam_labeled
views: /home/ww/vv/dataset/Replica/{scene}_s10_200f
v5 metrics: /home/ww/oviovo_experiments/20260722_route3_replica8_surface_consensus/evaluations_observation_v5/{scene}/metrics.json
```

Use fresh-directory atomic publication and SHA-256 every consumed manifest, snapshot member, cache manifest, config, and source module. Never overwrite v5 files.

- [ ] **Step 4: Implement the one-scene CLI**

Expose:

```bash
python scripts/evaluation/evaluate_oviv2_view_consensus.py \
  --config configs/evaluation/oviv2_view_consensus_replica8.json \
  --scene room0 \
  --variant V0 \
  --frame-limit 20 \
  --output /home/ww/oviovo_experiments/20260722_view_consensus/smoke_room0_v0
```

The CLI preflights exactly one cache record per requested frame, loads observations once, builds graph evidence once, emits requested pyramid strata, measures visibility, projects candidates, adds hybrid unions, composes the suffix, evaluates ordered AP, and writes:

```text
metrics.json
view_consensus_audit.json
proposal_summary.json
prefix_fingerprint.txt
```

The audit records every config value, source/input hash, observation/edge/proposal/rejection count, primary and suffix IDs, prefix fingerprint, runtime, peak RSS, and V0 parity result. GT paths are passed only to ordered evaluation after candidate construction has completed and candidate fingerprints have been written.

- [ ] **Step 5: Add CLI failure and determinism tests**

Test missing cache frames, corrupt payloads, feature-model mismatch, scene/revision/hash mismatch, non-finite config, existing output, candidate generation signature containing no `ground_truth`, exact semantic/F5 passthrough from v5, and two fresh runs producing identical candidate/prefix fingerprints.

- [ ] **Step 6: Run GREEN and commit**

```bash
pytest -q \
  tests/evaluation/test_oviv2_ordered_instance.py \
  tests/evaluation/test_oviv2_view_graph.py \
  tests/evaluation/test_oviv2_view_proposals.py \
  tests/evaluation/test_oviv2_view_consensus_io.py \
  tests/evaluation/test_evaluate_oviv2_view_consensus_cli.py
git add \
  src/evaluation/oviv2_view_consensus_io.py \
  scripts/evaluation/evaluate_oviv2_view_consensus.py \
  tests/evaluation/test_oviv2_view_consensus_io.py \
  tests/evaluation/test_evaluate_oviv2_view_consensus_cli.py
git commit -m "feat: evaluate audited OVIV2 view-consensus proposals"
```

Expected: all focused tests pass.

### Task 6: Bounded room0 Search And Frozen Configuration

**Files:**
- Create: `configs/evaluation/oviv2_view_consensus_replica8.json`
- Create: `scripts/evaluation/search_oviv2_view_consensus_room0.py`
- Create: `tests/evaluation/test_search_oviv2_view_consensus_room0.py`

- [ ] **Step 1: Write failing configuration and staged-selection tests**

The JSON must bind all eight scene roots and exact v5 baseline metrics. Test staged top-three selection, AP25 feasibility, AP50-first lexicographic order, suffix-count/runtime tie breaks, and rejection when V0 fails.

```python
winner = select_room0_winner(
    results,
    baseline_ap25=0.4577581330960257,
    baseline_ap50=0.16657105953750673,
)
assert winner["ap25"] >= 0.4577581330960257
assert winner["ap50"] == max(
    item["ap50"] for item in results if item["ap25"] >= 0.4577581330960257
)
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_search_oviv2_view_consensus_room0.py
```

Expected: search module and configuration are missing.

- [ ] **Step 3: Add the exact bounded grid**

Use the following staged grid, retaining the top three feasible configurations after each stage:

```json
{
  "V1": {
    "minimum_distinct_views": [2, 3, 4],
    "strong_voxel_iou": [0.05, 0.10, 0.20],
    "strong_directed_coverage": [0.20, 0.35]
  },
  "V2": {
    "core_vote_fraction": [0.40, 0.55, 0.70],
    "inclusive_coverage": [0.20, 0.35],
    "dilation_cells": [0, 1]
  },
  "V3": {
    "minimum_supporter_ratio": [0.50, 0.67, 0.80],
    "maximum_border_fraction": [0.25, 0.50],
    "depth_tolerance_m": [0.05, 0.10]
  },
  "V4": {
    "weak_merge_iou": [0.05, 0.10, 0.20]
  },
  "V5": {
    "hybrid_minimum_iou": [0.05, 0.15, 0.30],
    "suffix_deduplication_iou": [0.85, 0.90, 0.95]
  }
}
```

Fixed values are voxel size `0.05`, pixel stride `2`, minimum valid depth points `10`, minimum overlap voxels `4`, minimum feature cosine `0.20`, minimum proposal voxels `10`, maximum proposals `1024`, minimum projected vertices `100`, and V5 hybrid maximum IoU `0.90`.

- [ ] **Step 4: Implement staged in-process search**

Load observations and pair evidence once. For each stage, evaluate its Cartesian product against each retained parent config, write every candidate to a unique hash-named directory, reject AP25 below baseline, and sort by `(-ap50, -ap25, suffix_count, runtime_seconds, config_hash)`. Write `search_manifest.json`, `stage_leaderboard.json`, `frozen_config.json`, and SHA-256 bindings. The frozen result must strictly improve room0 AP50 and preserve semantic/F5 metrics byte-for-byte.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/evaluation/test_search_oviv2_view_consensus_room0.py
git add \
  configs/evaluation/oviv2_view_consensus_replica8.json \
  scripts/evaluation/search_oviv2_view_consensus_room0.py \
  tests/evaluation/test_search_oviv2_view_consensus_room0.py
git commit -m "feat: search and freeze room0 view consensus"
```

Expected: all search tests pass.

### Task 7: Per-Scene Promotion Gate

**Files:**
- Create: `scripts/evaluation/promote_oviv2_view_consensus.py`
- Create: `tests/evaluation/test_promote_oviv2_view_consensus.py`

- [ ] **Step 1: Write failing promotion tests**

Use eight synthetic metric packages. Require exact scene set, frozen config/source hashes, primary fingerprints, nondecreasing AP25/AP50 per scene within `1e-12`, strictly higher Replica-7 and Replica-8 AP50, and exact semantic/F5 equality.

```python
audit = promote(
    baseline_result=baseline_result,
    candidate_root=candidate_root,
    frozen_config=frozen_config,
    output=output,
)
assert audit["status"] == "PASS"
assert all(scene["status"] == "PASS" for scene in audit["scenes"].values())
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/evaluation/test_promote_oviv2_view_consensus.py
```

Expected: promotion script is missing.

- [ ] **Step 3: Implement strict promotion and aggregate output**

Compute macro metrics directly from scene JSON files; do not accept supplied aggregates. Check every artifact checksum and reject missing, duplicate, stale, non-finite, or mismatched records. Emit `promotion_audit.json` and `result.json` only after all gates pass. A failed candidate writes only a failed audit and never edits paper tables.

- [ ] **Step 4: Run GREEN and full regression**

```bash
pytest -q \
  tests/evaluation/test_oviv2_ordered_instance.py \
  tests/evaluation/test_oviv2_view_graph.py \
  tests/evaluation/test_oviv2_view_proposals.py \
  tests/evaluation/test_oviv2_view_consensus_io.py \
  tests/evaluation/test_evaluate_oviv2_view_consensus_cli.py \
  tests/evaluation/test_search_oviv2_view_consensus_room0.py \
  tests/evaluation/test_promote_oviv2_view_consensus.py
git diff --check
```

Expected: all focused tests pass and no whitespace errors exist.

- [ ] **Step 5: Commit**

```bash
git add \
  scripts/evaluation/promote_oviv2_view_consensus.py \
  tests/evaluation/test_promote_oviv2_view_consensus.py
git commit -m "feat: gate OVIV2 view consensus across Replica"
```

### Task 8: Execute room0, Freeze, Then Promote Replica-7/8

**Files:**
- Create after accepted runs: `docs/superpowers/reports/2026-07-22-oviv2-view-consensus-results.md`
- Modify only after PASS: `docs/paper/OVIV2_REPLICA_RESULTS.md`
- Modify only after PASS: `docs/paper/benchmark_tables_baselines.md`
- Modify only after PASS: `docs/paper/benchmark_tables_baselines.tex`

- [ ] **Step 1: Run V0 and 20-frame smoke**

```bash
python scripts/evaluation/evaluate_oviv2_view_consensus.py \
  --config configs/evaluation/oviv2_view_consensus_replica8.json \
  --scene room0 --variant V0 --frame-limit 20 \
  --output /home/ww/oviovo_experiments/20260722_view_consensus/smoke_room0_v0
```

Expected: V0 prefix parity passes, output is nonempty, no GT dependency appears in candidate provenance, and peak RSS stays below the configured 32 GiB limit.

- [ ] **Step 2: Run the full room0 staged search**

```bash
python scripts/evaluation/search_oviv2_view_consensus_room0.py \
  --config configs/evaluation/oviv2_view_consensus_replica8.json \
  --output /home/ww/oviovo_experiments/20260722_view_consensus/room0_search
```

Expected: frozen room0 AP25 is at least `0.4577581330960257`, AP50 is above `0.16657105953750673`, and mIoU/mAcc/f-mIoU/F@5cm equal v5.

- [ ] **Step 3: Reproduce the frozen room0 candidate**

```bash
python scripts/evaluation/evaluate_oviv2_view_consensus.py \
  --config configs/evaluation/oviv2_view_consensus_replica8.json \
  --scene room0 \
  --frozen-config /home/ww/oviovo_experiments/20260722_view_consensus/room0_search/frozen_config.json \
  --output /home/ww/oviovo_experiments/20260722_view_consensus/room0_reproduction
```

Expected: candidate fingerprint, prefix fingerprint, metrics, and audit configuration match the frozen search winner.

- [ ] **Step 4: Evaluate all seven held-out scenes without retuning**

```bash
for scene in room1 room2 office0 office1 office2 office3 office4; do
  python scripts/evaluation/evaluate_oviv2_view_consensus.py \
    --config configs/evaluation/oviv2_view_consensus_replica8.json \
    --scene "$scene" \
    --frozen-config /home/ww/oviovo_experiments/20260722_view_consensus/room0_search/frozen_config.json \
    --output "/home/ww/oviovo_experiments/20260722_view_consensus/replica8/$scene"
done
```

Expected: every scene preserves its v5 primary prefix and has AP25/AP50 no lower than v5.

- [ ] **Step 5: Run promotion and publish only on PASS**

```bash
python scripts/evaluation/promote_oviv2_view_consensus.py \
  --baseline-result docs/paper/results/oviv2/replica/20260722-route3-surface-observation-v5/result.json \
  --candidate-root /home/ww/oviovo_experiments/20260722_view_consensus/replica8 \
  --room0 /home/ww/oviovo_experiments/20260722_view_consensus/room0_reproduction \
  --frozen-config /home/ww/oviovo_experiments/20260722_view_consensus/room0_search/frozen_config.json \
  --output /home/ww/oviovo_experiments/20260722_view_consensus/promotion
```

Expected: per-scene and aggregate status `PASS`, Replica-7/8 AP50 strictly improves, and no other metric decreases.

- [ ] **Step 6: Record results and verify the paper package**

Write exact per-scene and aggregate values, deltas, hashes, runtime, candidate counts, accepted config, and failed ablations to the report. Update paper tables only if promotion passed, then run:

```bash
pytest -q
python scripts/evaluation/verify_benchmark_table_package.py
git diff --check
```

Expected: complete test suite and benchmark package verification pass.

- [ ] **Step 7: Commit accepted evidence**

```bash
git add \
  docs/superpowers/reports/2026-07-22-oviv2-view-consensus-results.md \
  docs/paper/OVIV2_REPLICA_RESULTS.md \
  docs/paper/benchmark_tables_baselines.md \
  docs/paper/benchmark_tables_baselines.tex
git commit -m "docs: publish OVIV2 view consensus results"
```

If promotion fails, commit only the diagnostic report and keep the v5 paper tables unchanged.
