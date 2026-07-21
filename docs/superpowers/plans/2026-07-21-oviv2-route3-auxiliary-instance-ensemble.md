# OVIV2 Route 3 Auxiliary Instance Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GT-free auxiliary instance proposal stream that fuses the frozen Route 1 component hypotheses with a separately mapped `sam_labeled` snapshot while leaving semantic and geometry heads byte-identical.

**Architecture:** The primary snapshot remains authoritative for semantic and geometry metrics. A focused ensemble module builds one auxiliary parent hypothesis per non-empty entity, scores it only from its own mesh support, projects primary and auxiliary hypotheses independently in the evaluator, and performs one score-ordered cross-source deduplication before class-agnostic AP. The auxiliary snapshot is accepted only with a matching run manifest and complete SHA-256 provenance.

**Tech Stack:** Python, NumPy, SciPy `cKDTree`, pytest, existing OVIV2 snapshot/meshing/evaluation contracts.

---

### Task 1: Preprojected Instance Evaluation

**Files:**
- Modify: `src/evaluation/oviv2_instance_head.py`
- Modify: `tests/evaluation/test_oviv2_instance_head.py`

- [ ] **Step 1: Write failing tests for preprojected evaluation**

Add tests proving that `evaluate_projected_instance_hypotheses` returns the same AP/recall/count contract as the existing evaluator, filters masks below `min_instance_vertices`, deduplicates across overlapping source-prefixed IDs, and has no mesh or proposal-generation dependency.

```python
result = evaluate_projected_instance_hypotheses(
    projected,
    ground_truth,
    instance_semantic_ids={4},
    min_instance_vertices=2,
    deduplication_iou_threshold=0.9,
)
assert result["prediction_hypothesis_ids"] == ["base:e1", "aux:e2"]
assert result["ap50"] == pytest.approx(1.0)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_instance_head.py -k preprojected
```

Expected: import/attribute failure because the function does not exist.

- [ ] **Step 3: Extract the existing post-projection logic**

Implement the public function with the existing supported-mask filter, score/ID greedy deduplication, GT-instance IoU matrix, AP25/AP50 matching, recall, counts, IDs, and confidences. Change `evaluate_instance_hypotheses` to project once and delegate without changing its output.

- [ ] **Step 4: Verify GREEN and regression compatibility**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_instance_head.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/oviv2_instance_head.py tests/evaluation/test_oviv2_instance_head.py
git commit -m "refactor: expose projected OVIV2 instance evaluation"
```

### Task 2: GT-Free Auxiliary Entity Head

**Files:**
- Create: `src/evaluation/oviv2_instance_ensemble.py`
- Create: `tests/evaluation/test_oviv2_instance_ensemble.py`

- [ ] **Step 1: Write failing configuration and scoring tests**

Cover finite/range validation, deterministic entity ordering, empty entity omission, source-prefixed IDs, normalized mesh-support scoring, clipping, read-only indices, and absence of any ground-truth parameter.

```python
config = AuxiliaryInstanceConfig(score_weight=6.0, support_exponent=1.5, score_bias=0.05)
hypotheses = build_auxiliary_entity_hypotheses(mesh, {7: 4, 9: 5}, config)
assert [item.hypothesis_id for item in hypotheses] == ["aux:e7:parent", "aux:e9:parent"]
assert all(0.0 <= item.score <= 1.0 for item in hypotheses)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_instance_ensemble.py
```

Expected: module import failure.

- [ ] **Step 3: Implement the minimal auxiliary head**

Define a frozen `AuxiliaryInstanceConfig`. Build one `InstanceHypothesis` per positive entity ID present in the auxiliary mesh. Compute `support = entity_vertex_count / maximum_entity_vertex_count` and `score = clip(score_weight * support ** support_exponent + score_bias, 0, 1)`. Do not read projected masks, target vertices, GT, semantic correctness, or evaluation matches.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_instance_ensemble.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/oviv2_instance_ensemble.py tests/evaluation/test_oviv2_instance_ensemble.py
git commit -m "feat: add GT-free auxiliary OVIV2 instance proposals"
```

### Task 3: Auditable Dual-Snapshot Evaluator

**Files:**
- Modify: `scripts/evaluation/evaluate_oviv2_instance_head.py`
- Modify: `tests/evaluation/test_evaluate_oviv2_instance_head_cli.py`
- Modify: `scripts/evaluation/compare_oviv2_pareto.py`
- Modify: `tests/evaluation/test_compare_oviv2_pareto.py`

- [ ] **Step 1: Write failing CLI and provenance tests**

Add a fixture run manifest with method `OVIV2`, matching scene/revision, benchmark/vocabulary hashes, algorithm hash, and every auxiliary snapshot component hash. Require snapshot and run manifest together. Test missing pairs, scene/hash mismatch, output freshness, deterministic counts, auxiliary provenance, and exact semantic/F5 equality with the primary-only run.

```python
command += [
    "--auxiliary-instance-snapshot", str(aux_snapshot),
    "--auxiliary-instance-run-manifest", str(aux_manifest),
    "--auxiliary-score-weight", "6",
    "--auxiliary-support-exponent", "1.5",
    "--auxiliary-score-bias", "0.05",
    "--deduplication-iou-threshold", "0.9",
]
assert metrics["protocol"]["auxiliary_instance_ensemble"]["upstream_algorithm_hash"]
```

Also require `composed` comparator mode to bind and validate `auxiliary_instance_ensemble.algorithm_hash` when the field is present.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/evaluation/test_evaluate_oviv2_instance_head_cli.py tests/evaluation/test_compare_oviv2_pareto.py
```

Expected: CLI arguments are unrecognized and composed provenance validation is incomplete.

- [ ] **Step 3: Implement auxiliary loading and one final deduplication**

Validate the manifest before deriving the auxiliary mesh. Use its own geometry/evidence/ownership/registry, the primary vocabulary, and the primary mesh threshold. Project both streams independently; concatenate them; call `evaluate_projected_instance_hypotheses` once with the frozen ensemble threshold. Keep semantic and geometry metrics from the primary mesh. Record primary/auxiliary raw and supported counts, snapshot checksums, run-manifest SHA-256, upstream algorithm hash, source hashes, normalized config, and a complete ensemble algorithm hash.

- [ ] **Step 4: Verify GREEN and full focused regression**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_instance_head.py tests/evaluation/test_oviv2_instance_ensemble.py tests/evaluation/test_evaluate_oviv2_instance_head_cli.py tests/evaluation/test_compare_oviv2_pareto.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/evaluate_oviv2_instance_head.py scripts/evaluation/compare_oviv2_pareto.py tests/evaluation/test_evaluate_oviv2_instance_head_cli.py tests/evaluation/test_compare_oviv2_pareto.py
git commit -m "feat: evaluate auditable OVIV2 auxiliary instance ensembles"
```

### Task 4: Freeze and Gate room0 Route 3

**Files:**
- Modify: `docs/superpowers/reports/2026-07-21-oviv2-route1-room0-progress.md`

- [ ] **Step 1: Run the full room0 evaluator into a fresh output**

Use the Route 1 semantic replay and geometry head, primary scores `multiplier=12`, `view exponent=2.5`, `semantic exponent=1.5`, final deduplication `0.9`, and auxiliary scores `weight=6`, `support exponent=1.5`, `bias=0.05`. Bind the Route 2 `sam_labeled` final snapshot and run manifest.

- [ ] **Step 2: Apply the instance Pareto gate**

Compare against `composed_frozen_6c9c104/metrics.json` in `instance` mode. Expected: AP25 and AP50 strictly improve; mIoU, mAcc, f-mIoU, and F5 remain IEEE-identical.

- [ ] **Step 3: Reproduce byte-identically**

Run the same command into a second fresh directory and compare metrics/audit SHA-256. Reject the candidate if metrics differ or provenance is incomplete.

- [ ] **Step 4: Record the accepted or rejected result**

Append exact metrics, deltas, configuration, hashes, counts, and gate status to the room0 progress report. Keep this as a separately labeled Route 3 result until one frozen auxiliary stream exists for every Replica scene.

- [ ] **Step 5: Verify and commit**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_instance_head.py tests/evaluation/test_oviv2_instance_ensemble.py tests/evaluation/test_evaluate_oviv2_instance_head_cli.py tests/evaluation/test_compare_oviv2_pareto.py
git diff --check
```

Expected: all tests pass and no whitespace errors.

```bash
git add docs/superpowers/reports/2026-07-21-oviv2-route1-room0-progress.md
git commit -m "docs: record OVIV2 Route 3 room0 ensemble"
```
