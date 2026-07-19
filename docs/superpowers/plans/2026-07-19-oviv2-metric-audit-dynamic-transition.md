# OVIV2 Metric Audit and Dynamic Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OVIV2 Table 1 instance AP genuinely class-agnostic, publish a versioned corrected static result, and add the first snapshot-native dynamic current-map evaluator.

**Architecture:** Headline class-agnostic instance AP operates only on projected entity IDs and never depends on semantic feature availability. Existing semantic-class-constrained AP remains an explicitly named diagnostic. Dynamic evaluation consumes immutable labeled snapshots plus evaluator-only intervention domains and never imports the dirty legacy mutable pipeline.

**Tech Stack:** Python 3.10, NumPy, SciPy cKDTree, pytest, OVIV2 labeled meshes and snapshots, existing result finalization/import tools.

---

### Task 1: Add a Failing Class-Agnostic AP Contract

**Files:**
- Modify: `tests/evaluation/test_oviv2_replica.py`

- [ ] **Step 1: Add a featureless-entity headline test**

```python
def test_headline_instance_ap_is_class_agnostic_and_needs_no_entity_info() -> None:
    vertices = np.column_stack((np.arange(100, dtype=np.float32), np.zeros((100, 2))))
    mesh = _mesh(vertices, np.zeros(100, dtype=np.int64), np.full(100, 10, dtype=np.int64))
    gt = _ground_truth(vertices, np.ones(100, dtype=np.int64), np.ones(100, dtype=np.int64))

    metrics = evaluate_replica_voxel_map(
        mesh,
        gt,
        [],
        valid_semantic_ids={1},
        instance_semantic_ids={1},
        min_instance_vertices=100,
    )

    assert metrics["ap25"] == pytest.approx(1.0)
    assert metrics["ap50"] == pytest.approx(1.0)
    assert metrics["instance"]["class_agnostic"]["predicted_instance_count"] == 1
    assert metrics["instance"]["semantic_class_constrained"]["ap25"] == 0.0
```

- [ ] **Step 2: Add a cross-class identity test**

Use two GT objects with the same raw instance ID but different semantic classes and two predicted entity IDs. Assert class-agnostic matching treats them as two distinct masks derived from `(semantic_id, instance_id)` GT identity while headline AP remains independent of predicted semantic labels.

- [ ] **Step 3: Run and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_oviv2_replica.py::test_headline_instance_ap_is_class_agnostic_and_needs_no_entity_info -q
```

Expected: FAIL because current headline AP filters out entity `10` when `entity_info` is absent.

---

### Task 2: Separate Class-Agnostic and Semantic Instance Metrics

**Files:**
- Modify: `src/evaluation/oviv2_replica.py`
- Modify: `tests/evaluation/test_oviv2_replica.py`

- [ ] **Step 1: Extract deterministic mask builders**

Add private helpers:

```python
def _ground_truth_instance_masks(
    ground_truth: ReplicaGroundTruth,
    instance_semantic_ids: set[int],
    min_instance_vertices: int,
) -> list[tuple[tuple[int, int], np.ndarray]]: ...

def _predicted_entity_masks(
    projected: ProjectedLabels,
    min_instance_vertices: int,
) -> list[tuple[float, int, np.ndarray]]: ...
```

GT identity is `(semantic_id, instance_id)` so repeated raw instance IDs across classes cannot collapse. Prediction confidence is mapped vertex area normalized by the largest predicted entity and ties break by entity ID.

- [ ] **Step 2: Implement class-agnostic metrics**

```python
def _class_agnostic_instance_metrics(...):
    gt_masks = _ground_truth_instance_masks(...)
    predictions = _predicted_entity_masks(...)
    ap25, recall25 = _instance_threshold(predictions, gt_masks, 0.25)
    ap50, recall50 = _instance_threshold(predictions, gt_masks, 0.50)
    return {
        "ap25": ap25,
        "ap50": ap50,
        "recall25": recall25,
        "recall50": recall50,
        "predicted_instance_count": len(predictions),
        "ground_truth_instance_count": len(gt_masks),
        "prediction_entity_ids": [entity_id for _, entity_id, _ in predictions],
    }
```

This function must not read `EntityEvaluationInfo`, semantic confidence, or accepted-view count.

- [ ] **Step 3: Rename the current implementation**

Rename `_instance_metrics` to `_semantic_class_constrained_instance_metrics`. Keep its behavior for diagnostics.

- [ ] **Step 4: Publish the nested contract**

Return:

```python
"instance": {
    "class_agnostic": class_agnostic,
    "semantic_class_constrained": semantic_constrained,
},
"ap25": class_agnostic["ap25"],
"ap50": class_agnostic["ap50"],
```

Add protocol fields:

```python
"headline_instance_protocol": "class_agnostic",
"semantic_instance_protocol": "diagnostic_only",
```

- [ ] **Step 5: Update existing assertions without weakening them**

Existing class-constrained tests must assert under `metrics["instance"]["semantic_class_constrained"]`. Add separate headline assertions where the fixtures permit exact calculation.

- [ ] **Step 6: Run focused tests and verify GREEN**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_oviv2_replica.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Tasks 1-2**

```bash
git add src/evaluation/oviv2_replica.py tests/evaluation/test_oviv2_replica.py
git commit -m "fix: make OVIV2 instance AP class agnostic"
```

---

### Task 3: Update CLI Audit Artifacts Without Breaking Consumers

**Files:**
- Modify: `scripts/evaluation/evaluate_oviv2_replica.py`
- Modify: `tests/evaluation/test_evaluate_oviv2_replica_cli.py`

- [ ] **Step 1: Add a failing artifact contract test**

Require the CLI to write:

```text
per_class_semantic.json
class_agnostic_instance_ap.json
semantic_class_instance_ap.json
```

Assert `metrics.json` headline `ap25/ap50` equals `instance.class_agnostic`, not the semantic diagnostic.

- [ ] **Step 2: Run and verify RED**

Expected: old `per_class_instance_ap.json` layout fails the new contract.

- [ ] **Step 3: Implement versioned audit outputs**

Write both instance reports with sorted JSON. Keep the old filename only if needed as a compatibility copy containing a deprecation field; do not let it obscure protocol identity.

- [ ] **Step 4: Run CLI tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py \
  tests/evaluation/test_oviv2_replica.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/evaluation/evaluate_oviv2_replica.py \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py
git commit -m "feat: disclose OVIV2 instance metric protocols"
```

---

### Task 4: Generate a Machine-Readable Static Protocol Audit

**Files:**
- Create: `scripts/evaluation/audit_oviv2_replica_protocol.py`
- Create: `tests/evaluation/test_audit_oviv2_replica_protocol.py`

- [ ] **Step 1: Write failing audit tests**

The audit consumes old/new metrics JSON and optional OVI-MAP official diagnostics. Assert it records:

- projection comparator and threshold;
- minimum GT/predicted region sizes;
- whether headline AP is class-agnostic;
- whether accepted-view filtering affects headline AP;
- old/new headline deltas;
- source hashes;
- `headline_compatible` boolean with explicit reasons.

It must reject missing protocol fields, NaN/Inf, or a claimed class-agnostic protocol whose headline path points to semantic metrics.

- [ ] **Step 2: Run and verify RED**

Expected: script import failure.

- [ ] **Step 3: Implement the audit**

Use JSON parsing and SHA-256 only. Do not import mapper modules or inspect GT annotations. Atomically write a sorted `protocol_audit.json`.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_audit_oviv2_replica_protocol.py -q
git add scripts/evaluation/audit_oviv2_replica_protocol.py \
  tests/evaluation/test_audit_oviv2_replica_protocol.py
git commit -m "feat: audit OVIV2 Replica metric compatibility"
```

---

### Task 5: Re-Evaluate Replica-8 and Publish a New OVIV2 Result

**Files:**
- Create externally: a new versioned evaluation root under `/home/ww/oviovo_baseline_runs/`
- Create: `docs/paper/results/oviv2/replica/20260719-s10-200f-class-agnostic-ap-v2/result.json`
- Modify only if values change: registry and derived benchmark tables

- [ ] **Step 1: Reuse immutable snapshots**

Run the corrected evaluator over the eight existing frozen snapshots. Do not rerun frontends or mapping unless snapshot hashes fail.

- [ ] **Step 2: Compare old and new metrics**

Generate `protocol_audit.json`. Require semantic and geometry metrics to remain byte-equivalent; only class-agnostic instance fields may change.

- [ ] **Step 3: Finalize a new result ID**

Use a run ID containing `class-agnostic-ap-v2`. Preserve `20260719-s10-200f` unchanged. Include old/new audit paths and hashes in raw outputs.

- [ ] **Step 4: Import after provenance verification**

Verify every source JSON pointer and hash, then update registry pointers only for OVIV2 tokens owned by the OVIV2 result.

- [ ] **Step 5: Commit scoped result changes**

Do not include dirty legacy mapping/pipeline files or the untracked benchmark snapshot document.

---

### Task 6: Define Snapshot-Native Dynamic Evaluation Contracts

**Files:**
- Create: `src/evaluation/oviv2_dynamic.py`
- Create: `tests/evaluation/test_oviv2_dynamic.py`

- [ ] **Step 1: Write failing immutable-domain tests**

Define the desired API in tests:

```python
@dataclass(frozen=True)
class DynamicGroundTruthFrame:
    frame_id: int
    intervention_id: str
    current: ReplicaGroundTruth
    removed_region_vertices_xyz: np.ndarray
    revealed_background_vertices_xyz: np.ndarray

@dataclass(frozen=True)
class DynamicSnapshotEvaluation:
    frame_id: int
    current_miou: float
    ghost_rate: float
    background_f5: float
```

Tests require read-only arrays, matching frame order, finite values, and rejection of negative or duplicate frame IDs.

- [ ] **Step 2: Run and verify RED**

Expected: module import failure.

- [ ] **Step 3: Implement one-frame metrics**

`evaluate_dynamic_snapshot(mesh, ground_truth, ...)` must:

- reuse the static semantic projection for `current_miou`;
- compute ghost rate as the fraction of predicted geometry within the removed-region tolerance domain;
- compute background F5 against revealed-background vertices using raw geometry distances;
- return explicit numerator/denominator audit counts.

All intervention data is evaluator-only.

- [ ] **Step 4: Add hand-computed perfect, stale, and empty fixtures**

Perfect current map: current mIoU/F5 `1`, ghost `0`.

Stale map retaining only removed geometry: ghost `1`, background F5 `0`.

Empty prediction: finite zeros.

- [ ] **Step 5: Run focused tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_oviv2_dynamic.py -q
git add src/evaluation/oviv2_dynamic.py tests/evaluation/test_oviv2_dynamic.py
git commit -m "feat: evaluate OVIV2 current-state snapshots"
```

---

### Task 7: Add Recovery-Frames Sequence Aggregation

**Files:**
- Modify: `src/evaluation/oviv2_dynamic.py`
- Modify: `tests/evaluation/test_oviv2_dynamic.py`

- [ ] **Step 1: Add a failing recovery test**

For evaluations with background F5 `[0.1, 0.6, 0.91, 0.94]` after intervention and threshold `0.9`, assert recovery frames is `2`. Require `None` plus `recovered=false` when the threshold is never reached.

- [ ] **Step 2: Verify RED**

Expected: missing aggregation API.

- [ ] **Step 3: Implement aggregation**

```python
def aggregate_dynamic_sequence(
    evaluations: Sequence[DynamicSnapshotEvaluation],
    *,
    intervention_frame_id: int,
    recovery_background_f5: float = 0.9,
) -> dict[str, Any]: ...
```

Return macro current mIoU, macro ghost rate, macro background F5, recovery frames, recovered flag, frame IDs, and threshold.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_oviv2_dynamic.py -q
git add src/evaluation/oviv2_dynamic.py tests/evaluation/test_oviv2_dynamic.py
git commit -m "feat: aggregate OVIV2 recovery metrics"
```

---

## Completion Gate

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q -p no:cacheprovider \
  tests/oviv2 \
  tests/evaluation/test_oviv2_replica.py \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py \
  tests/evaluation/test_audit_oviv2_replica_protocol.py \
  tests/evaluation/test_oviv2_dynamic.py \
  tests/evaluation/test_oviv2_result.py \
  tests/evaluation/test_import_benchmark_results.py \
  tests/test_benchmark_table_package.py
```

Then verify the new result hashes and run the 19-reference-repository workspace verifier. Completion requires a versioned class-agnostic OVIV2 result and snapshot-native dynamic metric fixtures without any dependency on legacy mutable map classes.
