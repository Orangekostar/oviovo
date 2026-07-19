# OVIV2 Stage 0/1 Precision Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify the current room0 association/semantic ceiling and replace OVIV2's permissive, early-label-locked entity backend with causal feature-aware assignment, correctable semantic memory, and owner-authoritative mesh labels.

**Architecture:** Keep the sparse TSDF, evidence, visibility, ownership, and evaluator contracts. Extend cached observations with normalized visual features, centralize candidate scoring and Hungarian assignment, maintain a bounded causal observation graph, persist correctable entity semantic state in snapshot schema v2, and derive object-voxel semantics from the current owner entity. Stage 0 remains evaluator-only and cannot feed ground truth into runtime mapping.

**Tech Stack:** Python 3.10, NumPy, SciPy `linear_sum_assignment`, Open3D, plyfile, pytest, JSON/JSONL/NPZ immutable artifacts.

---

## Scope And File Structure

This plan implements only Stage 0 and Stage 1 from the approved design. RADSeg, the dense semantic sliding window, and cross-layer fusion are deferred to a separate implementation plan after this backend passes room0.

Create:

- `src/evaluation/oviv2_diagnostics.py`: pure GT-domain diagnostic calculations.
- `scripts/evaluation/diagnose_oviv2_room0.py`: immutable Stage 0 diagnostic CLI.
- `tests/evaluation/test_oviv2_diagnostics.py`: hand-computed diagnostic fixtures.
- `tests/evaluation/test_diagnose_oviv2_room0.py`: CLI/provenance tests.
- `src/oviv2/semantic_memory.py`: sparse class posterior, prototype bank, and informative-view bank.
- `tests/oviv2/test_semantic_memory.py`: semantic-memory unit tests.
- `src/oviv2/association.py`: spatial gates, multimodal scores, deterministic Hungarian assignment.
- `tests/oviv2/test_association.py`: association unit tests.
- `src/oviv2/observation_graph.py`: bounded causal observation edges and third-view support.
- `tests/oviv2/test_observation_graph.py`: causality, expiry, support, and revocation tests.
- `configs/oviv2_replica_room0_precision_stage1.json`: isolated Stage 1 room0 configuration.

Modify:

- `src/oviv2/observations.py`: load and validate cached image/text features.
- `tests/oviv2/test_observations.py`: feature cache contract.
- `src/oviv2/tracking.py`: use joint assignment and the causal graph.
- `tests/oviv2/test_tracking.py`: adjacency, deterministic assignment, and relabel tests.
- `src/oviv2/entities.py`: correctable semantics, joint track/entity assignment, schema-v2 JSONL.
- `tests/oviv2/test_entities.py`: posterior correction, batch matching, and persistence.
- `src/oviv2/runtime.py`: batch resolution and object-semantic ownership authority.
- `tests/oviv2/test_runtime.py`: no stale object labels and runtime counters.
- `src/oviv2/meshing.py`: override owned object labels from current entities.
- `tests/oviv2/test_meshing.py`: owner-authoritative mesh fixtures.
- `src/oviv2/snapshot.py`: snapshot schema v2 with an embedded entity registry.
- `tests/oviv2/test_snapshot.py`: v1 compatibility and v2 round trip.
- `scripts/evaluation/evaluate_oviv2_replica.py`: consume embedded v2 entities, retain v1 fallback.
- `scripts/run_oviv2_replica.py`: feature provenance, new configuration, counters, and v2 mesh export.
- `src/oviv2/__init__.py`: export the new public value types.

Do not modify dirty legacy files under `src/modules/`, `src/pipelines/`, `src/core/`, or their existing tests.

### Task 1: Add Pure Stage 0 Diagnostics

**Files:**
- Create: `src/evaluation/oviv2_diagnostics.py`
- Create: `tests/evaluation/test_oviv2_diagnostics.py`

- [ ] **Step 1: Write failing semantic-oracle and merge/fragmentation tests**

```python
import numpy as np

from src.evaluation.oviv2_diagnostics import diagnose_projected_labels


def test_diagnostics_separates_semantic_error_from_entity_geometry() -> None:
    report = diagnose_projected_labels(
        predicted_semantic_ids=np.asarray([3, 3, 3, 3, 2, 2, 0]),
        predicted_entity_ids=np.asarray([10, 10, 10, 10, 11, 11, 0]),
        gt_semantic_ids=np.asarray([2, 2, 3, 3, 3, 3, 2]),
        gt_instance_ids=np.asarray([1, 1, 2, 2, 2, 2, 3]),
        valid_semantic_ids={2, 3},
        minimum_overlap_vertices=1,
        minimum_overlap_fraction=0.20,
    )

    assert report["headline_eligible"] is False
    assert report["entity_geometry"]["overmerged_entity_ids"] == [10]
    assert report["entity_geometry"]["fragmented_gt_instance_ids"] == [2]
    assert report["semantic"]["oracle_entity_labels"] == {"10": 2, "11": 3}
    assert report["semantic"]["oracle_miou"] > report["semantic"]["current_miou"]


def test_diagnostics_rejects_shape_and_domain_errors() -> None:
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        diagnose_projected_labels(
            np.asarray([1]),
            np.asarray([1, 2]),
            np.asarray([1]),
            np.asarray([1]),
            valid_semantic_ids={1},
        )
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_oviv2_diagnostics.py
```

Expected: collection fails with `ModuleNotFoundError: src.evaluation.oviv2_diagnostics`.

- [ ] **Step 3: Implement deterministic diagnostic calculations**

Implement these public functions in `src/evaluation/oviv2_diagnostics.py`:

```python
from __future__ import annotations

import numpy as np


def _ids(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    if np.any(result < 0):
        raise ValueError(f"{name} must be non-negative")
    return result


def semantic_iou_from_ids(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid_semantic_ids: set[int],
) -> dict[str, object]:
    predicted = _ids(predicted, "predicted")
    ground_truth = _ids(ground_truth, "ground_truth")
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground_truth must have the same shape")
    valid = sorted(int(value) for value in valid_semantic_ids)
    if not valid or valid[0] <= 0:
        raise ValueError("valid_semantic_ids must contain positive IDs")
    domain = np.isin(ground_truth, valid)
    predicted_domain = predicted[domain]
    ground_truth_domain = ground_truth[domain]
    per_class: dict[str, float] = {}
    for semantic_id in sorted(int(value) for value in np.unique(ground_truth_domain)):
        gt_mask = ground_truth_domain == semantic_id
        pred_mask = predicted_domain == semantic_id
        union = int(np.sum(gt_mask | pred_mask))
        per_class[str(semantic_id)] = (
            float(np.sum(gt_mask & pred_mask) / union) if union else 0.0
        )
    return {
        "miou": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "per_class_iou": per_class,
        "evaluated_vertex_count": int(np.sum(domain)),
    }


def diagnose_projected_labels(
    predicted_semantic_ids: np.ndarray,
    predicted_entity_ids: np.ndarray,
    gt_semantic_ids: np.ndarray,
    gt_instance_ids: np.ndarray,
    *,
    valid_semantic_ids: set[int],
    minimum_overlap_vertices: int = 100,
    minimum_overlap_fraction: float = 0.10,
) -> dict[str, object]:
    arrays = [
        _ids(predicted_semantic_ids, "predicted_semantic_ids"),
        _ids(predicted_entity_ids, "predicted_entity_ids"),
        _ids(gt_semantic_ids, "gt_semantic_ids"),
        _ids(gt_instance_ids, "gt_instance_ids"),
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all ID arrays must have the same shape")
    if minimum_overlap_vertices <= 0:
        raise ValueError("minimum_overlap_vertices must be positive")
    if not 0.0 <= minimum_overlap_fraction <= 1.0:
        raise ValueError("minimum_overlap_fraction must lie in [0, 1]")

    pred_semantic, pred_entity, gt_semantic, gt_instance = arrays
    current = semantic_iou_from_ids(pred_semantic, gt_semantic, valid_semantic_ids)
    oracle = np.zeros_like(pred_semantic)
    oracle_labels: dict[str, int] = {}
    significant_by_entity: dict[int, list[int]] = {}
    for entity_id in sorted(int(value) for value in np.unique(pred_entity) if value > 0):
        entity_mask = pred_entity == entity_id
        valid_labels = gt_semantic[entity_mask & np.isin(gt_semantic, sorted(valid_semantic_ids))]
        if valid_labels.size:
            labels, counts = np.unique(valid_labels, return_counts=True)
            order = np.lexsort((labels, -counts))
            winner = int(labels[order[0]])
            oracle[entity_mask] = winner
            oracle_labels[str(entity_id)] = winner
        entity_size = int(np.sum(entity_mask))
        significant: list[int] = []
        for instance_id in sorted(int(value) for value in np.unique(gt_instance[entity_mask]) if value > 0):
            intersection = int(np.sum(entity_mask & (gt_instance == instance_id)))
            if (
                intersection >= minimum_overlap_vertices
                and intersection / entity_size >= minimum_overlap_fraction
            ):
                significant.append(instance_id)
        significant_by_entity[entity_id] = significant

    entities_by_instance: dict[int, list[int]] = {}
    for entity_id, instance_ids in significant_by_entity.items():
        for instance_id in instance_ids:
            entities_by_instance.setdefault(instance_id, []).append(entity_id)
    return {
        "headline_eligible": False,
        "semantic": {
            "current_miou": current["miou"],
            "oracle_miou": semantic_iou_from_ids(
                oracle, gt_semantic, valid_semantic_ids
            )["miou"],
            "oracle_entity_labels": oracle_labels,
        },
        "entity_geometry": {
            "overmerged_entity_ids": sorted(
                entity_id for entity_id, values in significant_by_entity.items() if len(values) > 1
            ),
            "fragmented_gt_instance_ids": sorted(
                instance_id for instance_id, values in entities_by_instance.items() if len(values) > 1
            ),
            "significant_gt_instances_by_entity": {
                str(key): values for key, values in sorted(significant_by_entity.items())
            },
        },
    }
```

Use only positive predicted entity IDs and positive GT instance IDs. For each predicted entity, count GT-instance intersections. Mark an intersection significant when it meets both `minimum_overlap_vertices` and `minimum_overlap_fraction` of the predicted entity. An entity touching two significant GT instances is over-merged. Reverse the relation to find GT instances represented by two or more predicted entities. Assign each predicted entity its majority valid GT semantic ID for the semantic oracle, leaving unmatched vertices as zero. Sort every emitted ID list and JSON key deterministically.

- [ ] **Step 4: Run the focused tests**

Run the Task 1 pytest command. Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/evaluation/oviv2_diagnostics.py tests/evaluation/test_oviv2_diagnostics.py
git commit -m "feat: add OVIV2 association ceiling diagnostics"
```

### Task 2: Add The Immutable Room0 Diagnostic CLI

**Files:**
- Create: `scripts/evaluation/diagnose_oviv2_room0.py`
- Create: `tests/evaluation/test_diagnose_oviv2_room0.py`

- [ ] **Step 1: Write the failing CLI publication test**

```python
def test_diagnostic_cli_writes_non_headline_report_atomically(tmp_path, replica_fixture) -> None:
    output = tmp_path / "diagnostics.json"
    exit_code = main([
        "--pred-semantic", str(replica_fixture.pred_semantic),
        "--pred-entity", str(replica_fixture.pred_entity),
        "--gt-mesh", str(replica_fixture.gt_mesh),
        "--gt-info", str(replica_fixture.gt_info),
        "--manifest", str(replica_fixture.manifest),
        "--scene", "room0",
        "--output", str(output),
        "--minimum-overlap-vertices", "1",
    ])

    payload = json.loads(output.read_text())
    assert exit_code == 0
    assert payload["headline_eligible"] is False
    assert payload["scene"] == "room0"
    assert set(payload["input_sha256"]) == {
        "pred_semantic", "pred_entity", "gt_mesh", "gt_info", "manifest"
    }
```

Build the fixture with a four-vertex semantic PLY, matching `info_semantic.json`, a one-scene manifest, and two `.npy` prediction arrays. Add a second test proving an existing output is rejected unless `--overwrite` is supplied.

Use this concrete fixture payload:

```python
@dataclass(frozen=True)
class ReplicaFixture:
    pred_semantic: Path
    pred_entity: Path
    gt_mesh: Path
    gt_info: Path
    manifest: Path


@pytest.fixture
def replica_fixture(tmp_path: Path) -> ReplicaFixture:
    vertices = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices["x"] = [0.0, 1.0, 1.0, 0.0]
    vertices["y"] = [0.0, 0.0, 1.0, 1.0]
    faces = np.zeros(2, dtype=[("vertex_indices", "i4", (3,)), ("object_id", "i4")])
    faces["vertex_indices"] = [[0, 1, 2], [0, 2, 3]]
    faces["object_id"] = [0, 1]
    gt_mesh = tmp_path / "mesh_semantic.ply"
    PlyData([
        PlyElement.describe(vertices, "vertex"),
        PlyElement.describe(faces, "face"),
    ]).write(gt_mesh)
    gt_info = tmp_path / "info_semantic.json"
    gt_info.write_text(json.dumps({"objects": [
        {"id": 0, "class_name": "chair"},
        {"id": 1, "class_name": "table"},
    ]}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "manifest_id": "diagnostic-fixture",
        "dataset": "Replica",
        "vocabulary": {"classes": ["chair", "table"]},
        "aliases": {},
        "scenes": [{"scene": "room0"}],
    }), encoding="utf-8")
    pred_semantic = tmp_path / "semantic.npy"
    pred_entity = tmp_path / "entity.npy"
    np.save(pred_semantic, np.asarray([1, 1, 2, 2], dtype=np.int64))
    np.save(pred_entity, np.asarray([10, 10, 11, 11], dtype=np.int64))
    return ReplicaFixture(pred_semantic, pred_entity, gt_mesh, gt_info, manifest)
```

- [ ] **Step 2: Run the CLI tests and verify they fail**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_diagnose_oviv2_room0.py
```

Expected: collection fails because `diagnose_oviv2_room0` does not exist.

- [ ] **Step 3: Implement the CLI**

The CLI must:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-semantic", type=Path, required=True)
    parser.add_argument("--pred-entity", type=Path, required=True)
    parser.add_argument("--gt-mesh", type=Path, required=True)
    parser.add_argument("--gt-info", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-overlap-vertices", type=int, default=100)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest, scene = _load_manifest(args.manifest, args.scene)
    _verify_scene_inputs(scene, args.gt_mesh, args.gt_info)
    aliases = _normalized_aliases(manifest.get("aliases", {}))
    classes = [_normalize_label(value, aliases) for value in manifest["vocabulary"]["classes"]]
    class_to_id = {label: index + 1 for index, label in enumerate(classes)}
    ground_truth = load_replica_ground_truth(
        args.gt_mesh, args.gt_info, class_to_id=class_to_id, aliases=aliases
    )
    report = diagnose_projected_labels(
        np.load(args.pred_semantic, allow_pickle=False),
        np.load(args.pred_entity, allow_pickle=False),
        ground_truth.semantic_ids,
        ground_truth.instance_ids,
        valid_semantic_ids=set(class_to_id.values()),
        minimum_overlap_vertices=args.minimum_overlap_vertices,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
    )
    report.update({
        "scene": args.scene,
        "protocol": {
            "manifest_id": manifest.get("manifest_id"),
            "minimum_overlap_vertices": args.minimum_overlap_vertices,
            "minimum_overlap_fraction": args.minimum_overlap_fraction,
        },
        "input_sha256": {
            "pred_semantic": _sha256(args.pred_semantic),
            "pred_entity": _sha256(args.pred_entity),
            "gt_mesh": _sha256(args.gt_mesh),
            "gt_info": _sha256(args.gt_info),
            "manifest": _sha256(args.manifest),
        },
    })
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    _atomic_json(args.output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "output": str(args.output),
        "current_miou": report["semantic"]["current_miou"],
        "oracle_miou": report["semantic"]["oracle_miou"],
    }, sort_keys=True))
    return 0
```

Import `_load_manifest`, `_verify_scene_inputs`, `_normalized_aliases`, `_normalize_label`, and `load_replica_ground_truth` from `scripts/evaluation/evaluate_oviv2_replica.py`. Implement `_sha256` with 1 MiB chunks and `_atomic_json` with a temporary sibling file, `flush`, `os.fsync`, and `os.replace` before using the code above.

- [ ] **Step 4: Run CLI tests, then generate the real Stage 0 report**

Run the focused tests. Expected: all pass.

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/diagnose_oviv2_room0.py \
  --pred-semantic outputs/oviv2_replica8_frozen/room0/evaluation/gt_aligned_semantic_ids.npy \
  --pred-entity outputs/oviv2_replica8_frozen/room0/evaluation/gt_aligned_instance_ids.npy \
  --gt-mesh /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --manifest configs/evaluation/manifests/replica8.json \
  --scene room0 \
  --output outputs/oviv2_precision_stage0/room0_diagnostics.json
```

Expected: exit 0, `headline_eligible=false`, and finite current/oracle mIoU values.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/evaluation/diagnose_oviv2_room0.py tests/evaluation/test_diagnose_oviv2_room0.py
git commit -m "feat: publish OVIV2 room0 diagnostic audit"
```

### Task 3: Carry Cached Visual Features Into Observations

**Files:**
- Modify: `src/oviv2/observations.py:115-230`
- Modify: `tests/oviv2/test_observations.py:32-116`
- Modify: `tests/oviv2/test_tracking.py:9-32`
- Modify: `tests/oviv2/test_runtime.py:26-47`

- [ ] **Step 1: Extend cache fixtures and write failing feature tests**

Update `_write_cache` to optionally include `image_feats` and `text_feats`. Add:

```python
def test_adapter_loads_normalized_read_only_features(tmp_path, vocabulary) -> None:
    masks = np.ones((2, 4, 5), dtype=bool)
    image = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    text = np.asarray([[0.0, 5.0], [8.0, 6.0]], dtype=np.float32)
    _write_cache(
        tmp_path / "frame000000.pkl.gz",
        masks=masks,
        labels=["chair", "chair"],
        image_feats=image,
        text_feats=text,
    )
    adapter = CachedFrontendAdapter(tmp_path, vocabulary, min_valid_points=1,
                                    feature_model_id="clip-sha256:test")

    observations = adapter.observe(_frame(), 0)

    assert np.linalg.norm(observations[0].image_feature) == pytest.approx(1.0)
    assert np.linalg.norm(observations[1].text_feature) == pytest.approx(1.0)
    assert observations[0].image_feature.flags.writeable is False
    assert observations[0].feature_model_id == "clip-sha256:test"
    assert np.linalg.norm(observations[0].view_direction_xyz) == pytest.approx(1.0)
    assert observations[0].visible_pixel_count > 0
    assert 0.0 <= observations[0].border_contact_fraction <= 1.0
```

Also test mismatched feature counts, zero-norm rows, non-finite rows, and image/text dimension mismatch. Missing optional feature keys must remain valid and yield `None` fields.

- [ ] **Step 2: Run observation tests and verify the constructor/API failure**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_observations.py
```

Expected: fail because `CachedFrontendAdapter` lacks `feature_model_id` and `FrameObservation` lacks feature fields.

- [ ] **Step 3: Implement the feature contract**

Append optional fields to `FrameObservation`:

```python
image_feature: np.ndarray | None = None
text_feature: np.ndarray | None = None
feature_model_id: str | None = None
view_direction_xyz: tuple[float, float, float] | None = None
visible_pixel_count: int = 0
border_contact_fraction: float = 0.0
```

Normalize present vectors to contiguous read-only `float32` arrays in `__post_init__`. Require both vectors to be one-dimensional, finite, nonzero, and dimension-compatible when both exist. Require a non-empty `feature_model_id` whenever either feature exists.

Add `feature_model_id` to `CachedFrontendAdapter.__init__`. `_load` must read optional `(N, D)` `image_feats` and `text_feats`, validate the first dimension against `N`, and attach the corresponding row to every `FrameObservation`.

Compute `view_direction_xyz` as the normalized vector from `frame.pose[:3, 3]` to the lifted observation centroid. Compute `visible_pixel_count` from the mask and `border_contact_fraction` as the fraction of positive mask pixels on the four image borders. Validate all three derived fields in `FrameObservation.__post_init__`.

Update observation helpers in tracking/runtime tests to accept optional features without changing existing calls.

- [ ] **Step 4: Run observation, tracking, and runtime tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_observations.py tests/oviv2/test_tracking.py tests/oviv2/test_runtime.py
```

Expected: all pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/oviv2/observations.py tests/oviv2/test_observations.py \
  tests/oviv2/test_tracking.py tests/oviv2/test_runtime.py
git commit -m "feat: preserve cached OVIV2 visual features"
```

### Task 4: Implement Correctable Semantic Memory

**Files:**
- Create: `src/oviv2/semantic_memory.py`
- Create: `tests/oviv2/test_semantic_memory.py`
- Modify: `src/oviv2/__init__.py`

- [ ] **Step 1: Write failing posterior, prototype, and view-bank tests**

```python
def test_later_consistent_evidence_corrects_early_label() -> None:
    posterior = SparseClassPosterior.empty()
    posterior = posterior.update_label(2, confidence=0.9, quality=1.0)
    for _ in range(3):
        posterior = posterior.update_label(3, confidence=0.9, quality=1.0)
    assert posterior.best_semantic_id == 3
    assert posterior.margin > 0.0
    assert posterior.effective_support == pytest.approx(3.6)


def test_prototype_bank_keeps_distinct_hypotheses_and_merges_similar_views() -> None:
    bank = FeaturePrototypeBank(max_prototypes=3, merge_cosine=0.90)
    bank = bank.update((1.0, 0.0), "clip:a", quality=1.0)
    bank = bank.update((0.99, 0.01), "clip:a", quality=2.0)
    bank = bank.update((0.0, 1.0), "clip:a", quality=1.0)
    assert len(bank.prototypes) == 2
    assert bank.prototypes[0].support == pytest.approx(3.0)


def test_view_bank_rejects_redundant_low_quality_view() -> None:
    bank = InformativeViewBank(max_views=2, minimum_novelty_cosine=0.05)
    bank = bank.update(InformativeView(1, 1, 100, 0.8, (1.0, 0.0, 0.0)))
    bank = bank.update(InformativeView(2, 2, 80, 0.4, (1.0, 0.0, 0.0)))
    assert [view.observation_id for view in bank.views] == [1]
```

Add validation tests for invalid semantic IDs, confidence/quality, non-normalizable features, mixed model IDs, duplicate observation IDs, and bank capacity.

- [ ] **Step 2: Run tests and verify the missing module failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_semantic_memory.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement immutable semantic value types**

Implement:

```python
from dataclasses import dataclass, replace

import numpy as np


def _unit_tuple(value) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("feature must be a finite non-empty vector")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("feature must have nonzero norm")
    return tuple(float(item) for item in array / norm)


@dataclass(frozen=True)
class SparseClassPosterior:
    log_evidence: tuple[tuple[int, float], ...]
    effective_support: float

    @classmethod
    def empty(cls) -> "SparseClassPosterior":
        return cls((), 0.0)

    def update_label(self, semantic_id: int, confidence: float,
                     quality: float) -> "SparseClassPosterior":
        if semantic_id <= 0:
            raise ValueError("semantic_id must be positive")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must lie in [0, 1]")
        mass = float(confidence * quality)
        if mass == 0.0:
            return self
        values = dict(self.log_evidence)
        incoming = float(np.log(mass))
        values[semantic_id] = float(np.logaddexp(values.get(semantic_id, -np.inf), incoming))
        return SparseClassPosterior(
            tuple(sorted(values.items())), self.effective_support + mass
        )

    @property
    def probabilities(self) -> tuple[tuple[int, float], ...]:
        if not self.log_evidence:
            return ()
        logs = np.asarray([value for _, value in self.log_evidence], dtype=np.float64)
        shifted = np.exp(logs - float(np.max(logs)))
        normalized = shifted / float(np.sum(shifted))
        return tuple(
            (semantic_id, float(probability))
            for (semantic_id, _), probability in zip(self.log_evidence, normalized)
        )

    @property
    def best_semantic_id(self) -> int:
        return min(
            self.probabilities,
            key=lambda item: (-item[1], item[0]),
            default=(0, 0.0),
        )[0]

    @property
    def entropy(self) -> float:
        values = np.asarray([value for _, value in self.probabilities], dtype=np.float64)
        return float(-np.sum(values * np.log(np.maximum(values, 1e-12))))

    @property
    def margin(self) -> float:
        ranked = sorted((value for _, value in self.probabilities), reverse=True)
        return float(ranked[0] - ranked[1]) if len(ranked) > 1 else (ranked[0] if ranked else 0.0)

@dataclass(frozen=True)
class FeaturePrototype:
    vector: tuple[float, ...]
    model_id: str
    support: float
    observation_count: int

@dataclass(frozen=True)
class FeaturePrototypeBank:
    max_prototypes: int = 3
    merge_cosine: float = 0.90
    prototypes: tuple[FeaturePrototype, ...] = ()

    def update(self, feature, model_id: str, quality: float) -> "FeaturePrototypeBank":
        vector = _unit_tuple(feature)
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not np.isfinite(quality) or quality <= 0.0:
            raise ValueError("quality must be finite and positive")
        values = list(self.prototypes)
        compatible = [
            (float(np.dot(vector, item.vector)), index)
            for index, item in enumerate(values)
            if item.model_id == model_id and len(item.vector) == len(vector)
        ]
        best = max(compatible, default=(-1.0, -1))
        if best[0] >= self.merge_cosine:
            previous = values[best[1]]
            merged = np.asarray(previous.vector) * previous.support + np.asarray(vector) * quality
            values[best[1]] = FeaturePrototype(
                _unit_tuple(merged), model_id, previous.support + quality,
                previous.observation_count + 1,
            )
        else:
            incoming = FeaturePrototype(vector, model_id, quality, 1)
            if len(values) < self.max_prototypes:
                values.append(incoming)
            else:
                weakest = min(range(len(values)), key=lambda index: (values[index].support, index))
                if quality > values[weakest].support:
                    values[weakest] = incoming
        values.sort(key=lambda item: (-item.support, item.model_id, item.vector))
        return replace(self, prototypes=tuple(values))

    def maximum_cosine(self, feature, model_id: str) -> float | None:
        vector = _unit_tuple(feature)
        values = [
            float(np.dot(vector, item.vector)) for item in self.prototypes
            if item.model_id == model_id and len(item.vector) == len(vector)
        ]
        return max(values) if values else None

@dataclass(frozen=True)
class InformativeView:
    observation_id: int
    frame_id: int
    visible_pixel_count: int
    quality: float
    view_direction_xyz: tuple[float, float, float]

@dataclass(frozen=True)
class InformativeViewBank:
    max_views: int = 10
    minimum_novelty_cosine: float = 0.10
    views: tuple[InformativeView, ...] = ()

    def update(self, view: InformativeView) -> "InformativeViewBank":
        if any(item.observation_id == view.observation_id for item in self.views):
            return self
        values = list(self.views)
        direction = np.asarray(_unit_tuple(view.view_direction_xyz))
        similarities = [float(np.dot(direction, item.view_direction_xyz)) for item in values]
        if similarities:
            nearest = int(np.argmax(similarities))
            novelty = 1.0 - similarities[nearest]
            if novelty < self.minimum_novelty_cosine:
                if view.quality <= values[nearest].quality:
                    return self
                values[nearest] = view
            elif len(values) < self.max_views:
                values.append(view)
            else:
                weakest = min(range(len(values)), key=lambda index: (values[index].quality, index))
                if view.quality > values[weakest].quality:
                    values[weakest] = view
        else:
            values.append(view)
        values.sort(key=lambda item: (-item.quality, item.observation_id))
        return replace(self, views=tuple(values))
```

Use `np.logaddexp` to accumulate class evidence, normalize posterior probabilities with a stable log-sum-exp, and use deterministic `(quality, observation_id)` ordering for replacement ties. Store feature vectors as tuples of finite normalized floats so dataclass equality and JSON persistence are deterministic.

- [ ] **Step 4: Run semantic-memory tests and export the public classes**

Run the focused test. Expected: all pass. Add the six value types to `src/oviv2/__init__.py`, then run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_semantic_memory.py tests/oviv2/test_observations.py
```

- [ ] **Step 5: Commit Task 4**

```bash
git add src/oviv2/semantic_memory.py src/oviv2/__init__.py \
  tests/oviv2/test_semantic_memory.py
git commit -m "feat: add correctable OVIV2 semantic memory"
```

### Task 5: Implement Gated Multimodal Hungarian Assignment

**Files:**
- Create: `src/oviv2/association.py`
- Create: `tests/oviv2/test_association.py`

- [ ] **Step 1: Write failing gate, conflict, and joint-assignment tests**

```python
def target(
    target_id: int,
    keys: set[tuple[int, int, int]],
    *,
    bounds=None,
    semantic_id: int = 2,
    semantic_confidence: float = 0.9,
) -> AssociationTarget:
    points = np.asarray(sorted(keys), dtype=np.float64) * 0.05
    bounds_min, bounds_max = (
        (tuple(points.min(axis=0)), tuple(points.max(axis=0) + 0.05))
        if bounds is None else bounds
    )
    return AssociationTarget(
        target_id=target_id,
        voxel_keys=frozenset(keys),
        centroid_xyz=tuple(points.mean(axis=0)),
        bounds_min_xyz=tuple(float(value) for value in bounds_min),
        bounds_max_xyz=tuple(float(value) for value in bounds_max),
        semantic_id=semantic_id,
        semantic_confidence=semantic_confidence,
    )


def test_centroid_distance_without_spatial_support_cannot_match() -> None:
    config = AssociationConfig(bounds_expansion_m=0.0, max_centroid_distance_m=1.0)
    left = target(1, {(0, 0, 0)}, bounds=((0, 0, 0), (0.1, 0.1, 0.1)))
    right = target(2, {(2, 0, 0)}, bounds=((0.2, 0, 0), (0.3, 0.1, 0.1)))
    assert score_candidate(left, right, config) is None


def test_high_confidence_semantic_conflict_blocks_adjacent_merge() -> None:
    config = AssociationConfig(semantic_conflict_confidence=0.75)
    chair = target(1, {(0, 0, 0)}, semantic_id=2, semantic_confidence=0.9)
    table = target(2, {(0, 0, 0)}, semantic_id=3, semantic_confidence=0.9)
    result = score_candidate(chair, table, config)
    assert result is not None
    assert result.conflict is True
    assert result.accepted is False


def test_hungarian_assignment_is_one_to_one_and_deterministic() -> None:
    assignments = solve_assignment(
        (target(10, {(0, 0, 0)}), target(11, {(5, 0, 0)})),
        (target(20, {(0, 0, 0)}), target(21, {(5, 0, 0)})),
        AssociationConfig(),
    )
    assert [(item.left_id, item.right_id) for item in assignments] == [(10, 20), (11, 21)]
```

Add tests for feature-model mismatch, visual cosine contribution, directed overlap, expanded AABB support, score threshold rejection, and deterministic equal-score ties.

- [ ] **Step 2: Run tests and verify the missing module failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_association.py
```

- [ ] **Step 3: Implement association primitives**

Implement immutable `AssociationTarget`, `AssociationConfig`, `CandidateScore`, and `Assignment` values with the following fields and core functions:

```python
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.oviv2.addressing import VoxelKey


@dataclass(frozen=True)
class AssociationTarget:
    target_id: int
    voxel_keys: frozenset[VoxelKey]
    centroid_xyz: tuple[float, float, float]
    bounds_min_xyz: tuple[float, float, float]
    bounds_max_xyz: tuple[float, float, float]
    semantic_id: int = 0
    semantic_confidence: float = 0.0
    visual_feature: tuple[float, ...] | None = None
    feature_model_id: str | None = None
    free_space_conflict: bool = False


@dataclass(frozen=True)
class AssociationConfig:
    min_directed_overlap: float = 0.10
    bounds_expansion_m: float = 0.10
    max_centroid_distance_m: float = 0.60
    minimum_score: float = 0.45
    geometry_weight: float = 0.20
    overlap_weight: float = 0.25
    visual_weight: float = 0.30
    semantic_weight: float = 0.20
    temporal_weight: float = 0.05
    semantic_conflict_confidence: float = 0.75
    semantic_conflict_visual_override: float = 0.80


@dataclass(frozen=True)
class CandidateScore:
    left_id: int
    right_id: int
    score: float
    directed_overlap: float
    bounds_iou: float
    visual_cosine: float | None
    conflict: bool
    accepted: bool


@dataclass(frozen=True)
class Assignment:
    left_id: int
    right_id: int
    score: float


def directed_voxel_overlap(left: frozenset[VoxelKey], right: frozenset[VoxelKey]) -> float:
    return float(len(left & right) / len(left)) if left else 0.0


def expanded_bounds_iou(left: AssociationTarget, right: AssociationTarget,
                        expansion_m: float) -> float:
    left_min = np.asarray(left.bounds_min_xyz) - expansion_m
    left_max = np.asarray(left.bounds_max_xyz) + expansion_m
    right_min = np.asarray(right.bounds_min_xyz) - expansion_m
    right_max = np.asarray(right.bounds_max_xyz) + expansion_m
    intersection = np.maximum(0.0, np.minimum(left_max, right_max) - np.maximum(left_min, right_min))
    intersection_volume = float(np.prod(intersection))
    left_volume = float(np.prod(np.maximum(0.0, left_max - left_min)))
    right_volume = float(np.prod(np.maximum(0.0, right_max - right_min)))
    union = left_volume + right_volume - intersection_volume
    return intersection_volume / union if union > 0.0 else 0.0


def score_candidate(left: AssociationTarget, right: AssociationTarget,
                    config: AssociationConfig) -> CandidateScore | None:
    overlap = max(
        directed_voxel_overlap(left.voxel_keys, right.voxel_keys),
        directed_voxel_overlap(right.voxel_keys, left.voxel_keys),
    )
    bounds_iou = expanded_bounds_iou(left, right, config.bounds_expansion_m)
    if overlap < config.min_directed_overlap and bounds_iou <= 0.0:
        return None
    distance = float(np.linalg.norm(np.asarray(left.centroid_xyz) - right.centroid_xyz))
    if distance > config.max_centroid_distance_m and overlap < config.min_directed_overlap:
        return None
    geometry = max(0.0, 1.0 - distance / config.max_centroid_distance_m)
    visual: float | None = None
    if (
        left.visual_feature is not None and right.visual_feature is not None
        and left.feature_model_id == right.feature_model_id
        and len(left.visual_feature) == len(right.visual_feature)
    ):
        visual = float(np.dot(left.visual_feature, right.visual_feature))
    same_semantic = left.semantic_id > 0 and left.semantic_id == right.semantic_id
    semantic = 1.0 if same_semantic else (0.5 if 0 in {left.semantic_id, right.semantic_id} else 0.0)
    conflict = bool(
        left.free_space_conflict or right.free_space_conflict
        or (
            left.semantic_id > 0 and right.semantic_id > 0
            and left.semantic_id != right.semantic_id
            and min(left.semantic_confidence, right.semantic_confidence)
                >= config.semantic_conflict_confidence
            and (visual is None or visual < config.semantic_conflict_visual_override)
        )
    )
    components = [
        (config.geometry_weight, geometry),
        (config.overlap_weight, max(overlap, bounds_iou)),
        (config.semantic_weight, semantic),
        (config.temporal_weight, 1.0),
    ]
    if visual is not None:
        components.append((config.visual_weight, max(0.0, visual)))
    denominator = sum(weight for weight, _ in components)
    score = sum(weight * value for weight, value in components) / denominator
    return CandidateScore(
        left.target_id, right.target_id, float(score), overlap, bounds_iou, visual,
        conflict, bool(not conflict and score >= config.minimum_score),
    )


def solve_assignment(left, right, config: AssociationConfig) -> tuple[Assignment, ...]:
    left_values = tuple(sorted(left, key=lambda value: value.target_id))
    right_values = tuple(sorted(right, key=lambda value: value.target_id))
    if not left_values or not right_values:
        return ()
    costs = np.full((len(left_values), len(right_values)), 1e6, dtype=np.float64)
    scores: dict[tuple[int, int], CandidateScore] = {}
    for row, left_item in enumerate(left_values):
        for column, right_item in enumerate(right_values):
            candidate = score_candidate(left_item, right_item, config)
            if candidate is not None and candidate.accepted:
                scores[(row, column)] = candidate
                costs[row, column] = 1.0 - candidate.score + (row * len(right_values) + column) * 1e-12
    rows, columns = linear_sum_assignment(costs)
    result = [
        Assignment(scores[(row, column)].left_id, scores[(row, column)].right_id,
                   scores[(row, column)].score)
        for row, column in zip(rows, columns)
        if (row, column) in scores
    ]
    return tuple(sorted(result, key=lambda value: (value.left_id, value.right_id)))
```

A spatial candidate requires directed overlap at or above the configured threshold or positive expanded-bounds IoU; centroid distance is only a score after that gate. Reject free-space conflicts when supplied. Renormalize score weights over available components so missing visual features do not receive a zero penalty. A high-confidence label conflict is rejected unless compatible visual cosine exceeds `semantic_conflict_visual_override`. Use `scipy.optimize.linear_sum_assignment`, fill invalid pairs with a finite large cost, filter assignments below `minimum_score`, and add an ID-derived epsilon for deterministic ties.

- [ ] **Step 4: Run association tests**

Run the focused tests. Expected: all pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/oviv2/association.py tests/oviv2/test_association.py
git commit -m "feat: add gated OVIV2 joint association"
```

### Task 6: Add The Bounded Causal Observation Graph

**Files:**
- Create: `src/oviv2/observation_graph.py`
- Create: `tests/oviv2/test_observation_graph.py`

- [ ] **Step 1: Write failing causality and third-view tests**

```python
def node(frame_id: int, observation_id: int) -> ObservationNode:
    return ObservationNode(observation_id=observation_id, frame_id=frame_id)


def test_graph_rejects_future_edges_and_expires_old_frames() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(10, (node(10, 1),))
    with pytest.raises(ValueError, match="future"):
        graph.add_edge(ObservationEdge(1, 2, source_frame_id=10, target_frame_id=11, score=0.8))
    graph.add_frame(11, (node(11, 2),))
    graph.add_frame(12, (node(12, 3),))
    graph.add_frame(13, (node(13, 4),))
    assert graph.observation_ids == (2, 3, 4)


def test_ambiguous_edge_requires_independent_third_view_support() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(0, (node(0, 1),))
    graph.add_frame(1, (node(1, 2),))
    graph.add_edge(ObservationEdge(1, 2, 0, 1, 0.62))
    graph.add_frame(2, (node(2, 3),))
    graph.add_edge(ObservationEdge(1, 3, 0, 2, 0.80))
    graph.add_edge(ObservationEdge(2, 3, 1, 2, 0.78))
    assert graph.edge_is_supported(1, 2, ambiguous_below=0.70,
                                   third_view_min_score=0.75) is True


def test_explicit_revocation_removes_edge() -> None:
    graph = CausalObservationGraph(window_size=3)
    graph.add_frame(0, (node(0, 1),))
    graph.add_frame(1, (node(1, 2),))
    graph.add_edge(ObservationEdge(1, 2, 0, 1, 0.8))
    assert graph.remove_edge(1, 2) is True
    assert graph.edge_is_supported(1, 2, ambiguous_below=0.7,
                                   third_view_min_score=0.75) is False
```

Add a test that removing an expired supporting node revokes an ambiguous edge and a test that strong edges do not require third-view support.

- [ ] **Step 2: Run tests and verify the missing module failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_observation_graph.py
```

- [ ] **Step 3: Implement causal graph values and expiry**

Implement:

```python
@dataclass(frozen=True)
class ObservationNode:
    observation_id: int
    frame_id: int

@dataclass(frozen=True)
class ObservationEdge:
    left_observation_id: int
    right_observation_id: int
    source_frame_id: int
    target_frame_id: int
    score: float

class CausalObservationGraph:
    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self._nodes: dict[int, ObservationNode] = {}
        self._frame_ids: list[int] = []
        self._edges: dict[tuple[int, int], ObservationEdge] = {}

    @property
    def observation_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._nodes))

    def add_frame(self, frame_id: int, nodes: tuple[ObservationNode, ...]) -> None:
        if self._frame_ids and frame_id <= self._frame_ids[-1]:
            raise ValueError("frame IDs must increase monotonically")
        if any(node.frame_id != frame_id for node in nodes):
            raise ValueError("all nodes must belong to the added frame")
        if any(node.observation_id in self._nodes for node in nodes):
            raise ValueError("observation IDs must be globally unique in the active graph")
        self._frame_ids.append(frame_id)
        self._nodes.update((node.observation_id, node) for node in nodes)
        while len(self._frame_ids) > self.window_size:
            expired_frame = self._frame_ids.pop(0)
            expired_ids = {
                observation_id for observation_id, node in self._nodes.items()
                if node.frame_id == expired_frame
            }
            for observation_id in expired_ids:
                del self._nodes[observation_id]
            self._edges = {
                key: edge for key, edge in self._edges.items()
                if not expired_ids.intersection(key)
            }

    def add_edge(self, edge: ObservationEdge) -> None:
        if not self._frame_ids or edge.target_frame_id > self._frame_ids[-1]:
            raise ValueError("edge cannot reference a future frame")
        endpoints = {edge.left_observation_id, edge.right_observation_id}
        if not endpoints.issubset(self._nodes):
            raise ValueError("edge endpoints must be active observations")
        key = tuple(sorted(endpoints))
        previous = self._edges.get(key)
        if previous is None or edge.score > previous.score:
            self._edges[key] = edge

    def remove_edge(self, left_id: int, right_id: int) -> bool:
        return self._edges.pop(tuple(sorted((left_id, right_id))), None) is not None

    def edge_is_supported(self, left_id: int, right_id: int, *,
                          ambiguous_below: float, third_view_min_score: float) -> bool:
        key = tuple(sorted((left_id, right_id)))
        edge = self._edges.get(key)
        if edge is None:
            return False
        if edge.score >= ambiguous_below:
            return True
        candidates = set(self._nodes) - {left_id, right_id}
        for third_id in candidates:
            left_edge = self._edges.get(tuple(sorted((left_id, third_id))))
            right_edge = self._edges.get(tuple(sorted((right_id, third_id))))
            if (
                left_edge is not None and right_edge is not None
                and left_edge.score >= third_view_min_score
                and right_edge.score >= third_view_min_score
            ):
                return True
        return False
```

Keep only the latest `window_size` distinct frame IDs. Delete all incident edges when a node expires. Third-view support requires a distinct node connected to both endpoints with scores at or above the configured threshold. Never add an edge whose target frame is newer than the graph's latest committed frame.

- [ ] **Step 4: Run observation graph tests**

Expected: all pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/oviv2/observation_graph.py tests/oviv2/test_observation_graph.py
git commit -m "feat: add causal OVIV2 observation graph"
```

### Task 7: Refactor Local Tracking To Joint Assignment

**Files:**
- Modify: `src/oviv2/tracking.py:11-170`
- Modify: `tests/oviv2/test_tracking.py`

- [ ] **Step 1: Replace the old permissive behavior tests**

Add these tests while preserving bounded-window and expiry tests:

```python
def test_tracker_does_not_merge_adjacent_conflicting_objects() -> None:
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    chair = observation(0, {(0, 0, 20), (1, 0, 20)}, semantic_id=2, label="chair",
                        observation_id=1, image_feature=(1.0, 0.0))
    table = observation(0, {(1, 0, 20), (2, 0, 20)}, semantic_id=3, label="table",
                        observation_id=2, image_feature=(0.0, 1.0))
    first = tracker.update((chair, table), 0)
    next_chair = replace(chair, frame_id=1, timestamp=1.0, observation_id=101)
    next_table = replace(table, frame_id=1, timestamp=1.0, observation_id=102)
    second = tracker.update((next_chair, next_table), 1)
    assert len({track.track_id for track in second.accepted}) == 2


def test_tracker_joint_assignment_is_independent_of_observation_order() -> None:
    def run(reverse_second_frame: bool):
        tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
        left = observation(0, {(0, 0, 20)}, observation_id=1, image_feature=(1.0, 0.0))
        right = observation(0, {(10, 0, 20)}, observation_id=2, image_feature=(0.0, 1.0))
        tracker.update((left, right), 0)
        second = (
            replace(left, frame_id=1, timestamp=1.0, observation_id=101),
            replace(right, frame_id=1, timestamp=1.0, observation_id=102),
        )
        tracker.update(tuple(reversed(second)) if reverse_second_frame else second, 1)
        return tuple(
            (track.track_id, track.last_frame_id, tuple(sorted(track.voxel_keys)))
            for track in sorted(tracker.tracks.values(), key=lambda value: value.track_id)
        )

    assert run(False) == run(True)
```

Change the old `test_tracker_matches_geometry_before_semantic_label` expectation: a label change may remain on the same track only when visual evidence is compatible and the semantic conflict override passes.

- [ ] **Step 2: Run tracking tests and verify old behavior fails**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_tracking.py
```

Expected: at least the adjacent-object test fails because `_match` accepts overlap or centroid distance greedily.

- [ ] **Step 3: Integrate association and graph into `LocalTracker`**

Add `association: AssociationConfig`, `ambiguous_edge_score`, and `third_view_min_score` to `LocalTrackerConfig`. Extend `LocalTrackBatch` with integer `match_count`, `new_track_count`, `conflict_count`, and `revoked_edge_count` fields. At the start of `update`, insert all current object observations as graph nodes. Convert active tracks and observations to `AssociationTarget` values, evaluate candidate conflicts for the diagnostic counter, and call `solve_assignment` once per frame. For each proposed assignment, add a provisional edge between the current observation and the matched track's latest observation. Accept scores at or above `ambiguous_edge_score` directly; for lower accepted scores call `edge_is_supported`. Remove unsupported provisional edges, increment `revoked_edge_count`, and treat their observations as unmatched. Update matched tracks only after all assignments are decided, then create new tracks for unmatched observations. Keep `LocalTrack.observations` bounded and derive its visual feature as the normalized quality-weighted mean of compatible observation features.

Delete `_match`. Preserve monotonically increasing track IDs and deterministic output ordering by observation ID.

- [ ] **Step 4: Run tracking plus observation/association tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_observations.py tests/oviv2/test_association.py \
  tests/oviv2/test_observation_graph.py tests/oviv2/test_tracking.py
```

Expected: all pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add src/oviv2/tracking.py tests/oviv2/test_tracking.py
git commit -m "feat: use joint causal OVIV2 local tracking"
```

### Task 8: Add Correctable Entity Semantics And Batch Resolution

**Files:**
- Modify: `src/oviv2/entities.py:15-202`
- Modify: `tests/oviv2/test_entities.py`

- [ ] **Step 1: Write failing correction, batch, and schema tests**

```python
def test_entity_label_can_change_after_stronger_new_views() -> None:
    registry = EntityRegistry(EntityRegistryConfig(min_voxel_overlap=0.25))
    entity = registry.resolve(_track(0, {(0, 0, 20)}, semantic_id=2, label="chair"), 1)
    for revision in range(2, 5):
        entity = registry.resolve(
            _track(revision, {(0, 0, 20)}, semantic_id=3, label="stool"), revision
        )
    assert entity.semantic_id == 3
    assert entity.label == "stool"
    assert entity.semantic_margin > 0.0


def test_batch_resolution_prevents_two_tracks_claiming_one_entity() -> None:
    registry = EntityRegistry(EntityRegistryConfig(min_voxel_overlap=0.25))
    registry.resolve(_track(0, {(0, 0, 20), (1, 0, 20)}), revision=1)
    tracker = LocalTracker(LocalTrackerConfig(confirm_hits=1))
    tracks = tracker.update((
        observation(1, {(0, 0, 20), (1, 0, 20)}, observation_id=1),
        observation(1, {(1, 0, 20), (2, 0, 20)}, observation_id=2),
    ), 1).accepted
    resolved = registry.resolve_batch(tracks, revision=2)
    assert len({entity.entity_id for entity in resolved}) == 2


def test_v1_entity_jsonl_loads_as_correctable_v2_state(tmp_path) -> None:
    path = tmp_path / "entities.jsonl"
    path.write_text(json.dumps({
        "entity_id": 1,
        "semantic_id": 2,
        "label": "chair",
        "semantic_support": 0.9,
        "accepted_view_count": 1,
        "accepted_frame_ids": [0],
        "first_frame_id": 0,
        "last_frame_id": 0,
        "last_revision": 1,
        "lifecycle_state": "active",
        "voxel_keys": [[0, 0, 20]],
        "centroid_xyz": [0.0, 0.0, 1.0],
        "bounds_min_xyz": [0.0, 0.0, 1.0],
        "bounds_max_xyz": [0.0, 0.0, 1.0]
    }) + "\n", encoding="utf-8")
    entity = EntityRegistry.load(path).entities[1]
    assert entity.semantic_posterior.best_semantic_id == entity.semantic_id
    assert entity.feature_bank.prototypes == ()
```

Add round-trip tests for prototypes, informative views, posterior entropy/margin, and accepted observation IDs. Verify the JSONL has `schema_version: 2` and no point cloud state.

- [ ] **Step 2: Run entity tests and verify early-label/batch failures**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_entities.py
```

- [ ] **Step 3: Replace scalar support and implement `resolve_batch`**

Replace `semantic_support` with:

```python
semantic_posterior: SparseClassPosterior
feature_bank: FeaturePrototypeBank
view_bank: InformativeViewBank
accepted_observation_ids: frozenset[int]
semantic_entropy: float
semantic_margin: float
```

Only observations whose IDs are absent from `accepted_observation_ids` update semantic state. Compute view quality as `confidence * (1 - border_contact_fraction) * min(1, visible_pixel_count / 4096)`, then multiply by `min(1, len(voxel_keys) / 256)` and clamp to `[0, 1]`. Update current `semantic_id` and `label` from the posterior winner after every accepted observation. Insert `view_direction_xyz`, pixel count, and quality into the informative-view bank.

Add `EntityRegistry.resolve_batch(self, tracks: tuple[LocalTrack, ...], revision: int) -> tuple[PersistentEntity, ...]` and `EntityRegistry.semantic_labels(self) -> dict[int, tuple[int, float]]`. The latter returns `{entity_id: (semantic_id, winning_probability)}` for active and dormant entities with a positive posterior winner.

`resolve_batch` validates that every track is confirmed, builds one `AssociationTarget` per track and existing entity, calls the shared Hungarian solver once, applies matched updates in track-ID order, and then allocates one distinct entity for every unmatched track. Use private `_create_entity(track, revision)` and `_update_entity(previous, track, revision)` methods; both must consume only previously unseen observation IDs and return a complete immutable `PersistentEntity`. Retain `resolve` as `return self.resolve_batch((track,), revision)[0]`.

Write a first schema-v2 JSONL metadata record with `record_type="registry"`, the exact `EntityRegistryConfig`, and `schema_version=2`; write following records with `record_type="entity"` and explicit nested posterior/prototype/view payloads. Load schema-v1 files that contain only entity records by seeding the posterior from `semantic_id` and `semantic_support`; never invent a feature vector. Update evaluator entity-info loading to skip the registry metadata record.

- [ ] **Step 4: Run entity, semantic-memory, and association tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_semantic_memory.py tests/oviv2/test_association.py \
  tests/oviv2/test_entities.py
```

Expected: all pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add src/oviv2/entities.py tests/oviv2/test_entities.py
git commit -m "feat: make OVIV2 entity semantics correctable"
```

### Task 9: Make Current Owner Semantics Authoritative

**Files:**
- Modify: `src/oviv2/runtime.py:20-249`
- Modify: `src/oviv2/meshing.py:127-178`
- Modify: `tests/oviv2/test_runtime.py`
- Modify: `tests/oviv2/test_meshing.py`

- [ ] **Step 1: Write failing stale-label correction tests**

```python
def precision_runtime(*, confirm_hits: int) -> Oviv2Runtime:
    return Oviv2Runtime(
        "room0",
        Oviv2RuntimeConfig(tracker=LocalTrackerConfig(confirm_hits=confirm_hits)),
    )


def object_observation(frame_id: int, key, semantic_id: int, label: str) -> FrameObservation:
    value = observation(frame_id, ObservationKind.OBJECT, semantic_id, {key})
    return replace(value, label=label)


def test_owned_voxels_follow_current_entity_label_without_stale_votes() -> None:
    runtime = precision_runtime(confirm_hits=1)
    key = (0, 0, 20)
    runtime.process_frame(frame(0), (object_observation(0, key, 2, "chair"),))
    runtime.process_frame(frame(1), (object_observation(1, key, 3, "stool"),))
    runtime.process_frame(frame(2), (object_observation(2, key, 3, "stool"),))
    runtime.process_frame(frame(3), (object_observation(3, key, 3, "stool"),))

    mesh = derive_labeled_mesh(
        runtime.geometry, runtime.evidence, runtime.ownership,
        entity_semantics=runtime.registry.semantic_labels(),
    )
    owned = mesh.entity_ids == 1
    assert np.all(mesh.semantic_ids[owned] == 3)
    assert runtime.evidence.semantic_candidates(key) == ()


def test_structure_semantics_remain_voxel_evidence_without_owner() -> None:
    runtime = precision_runtime(confirm_hits=1)
    key = (0, 0, 20)
    runtime.process_frame(
        frame(0),
        (observation(0, ObservationKind.STRUCTURE, 1, {key}),),
    )
    mesh = derive_labeled_mesh(
        runtime.geometry, runtime.evidence, runtime.ownership,
        entity_semantics=runtime.registry.semantic_labels(),
    )
    assert np.any(mesh.semantic_ids == 1)
    assert np.all(mesh.entity_ids[mesh.semantic_ids == 1] == 0)
```

Add a test that an owner release removes the object semantic label unless structure evidence exists at that voxel. Add runtime assertions for assignment/conflict/new-entity counters in `RuntimeFrameResult`.

- [ ] **Step 2: Run runtime and meshing tests and verify stale evidence failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_runtime.py tests/oviv2/test_meshing.py
```

- [ ] **Step 3: Implement batch runtime and owner-authoritative meshing**

In `process_frame`, call `registry.resolve_batch(track_batch.accepted, next_revision)` once. Object observations update only entity evidence and ownership; they no longer copy entity labels into `SparseEvidenceStore`. Structure observations continue to update semantic voxel evidence.

Extend `derive_labeled_mesh` with the keyword-only argument `entity_semantics: Mapping[int, tuple[int, float]] | None = None`, after `weight_threshold`, while retaining the existing return type and positional parameters.

First read structure/unknown semantic evidence. If a current owner exists and `entity_semantics` has a positive semantic ID for that owner, override the vertex semantic ID and confidence. If an owner has no current semantic label, leave the vertex unknown unless independent structure evidence exists. Do not mutate any source store.

Extend `RuntimeFrameResult` with `matched_entity_count`, `new_entity_count`, `association_conflict_count`, and `revoked_edge_count`. Compute new versus matched entities by snapshotting `set(registry.entities)` before `resolve_batch`; source conflict and revocation counts from `LocalTrackBatch`.

- [ ] **Step 4: Run all OVIV2 core tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/oviv2
```

Expected: all pass.

- [ ] **Step 5: Commit Task 9**

```bash
git add src/oviv2/runtime.py src/oviv2/meshing.py \
  tests/oviv2/test_runtime.py tests/oviv2/test_meshing.py
git commit -m "feat: derive OVIV2 semantics from current owners"
```

### Task 10: Persist Entity Semantics In Snapshot Schema V2

**Files:**
- Modify: `src/oviv2/snapshot.py:19-208`
- Modify: `src/oviv2/runtime.py:235-249`
- Modify: `tests/oviv2/test_snapshot.py`
- Modify: `scripts/evaluation/evaluate_oviv2_replica.py:270-313`

- [ ] **Step 1: Write failing v2 and v1 compatibility tests**

```python
def test_v2_snapshot_embeds_registry_and_hashes_entities(tmp_path) -> None:
    metadata, geometry, evidence, ownership = _components()
    metadata = replace(metadata, schema_version=2)
    registry = EntityRegistry()
    original = registry.resolve(_track(0, {(0, 0, 20)}), revision=1)
    registry.entities = {11: replace(original, entity_id=11)}
    registry._next_entity_id = 12
    snapshot = VoxelMapSnapshot.commit(
        tmp_path / "snapshot", metadata, geometry, evidence, ownership, registry=registry
    )
    restored = VoxelMapSnapshot.load(snapshot.path)
    assert restored.metadata.schema_version == 2
    assert restored.registry.entities == registry.entities
    assert "entities.jsonl" in restored.checksums


def test_v1_snapshot_remains_readable_without_registry(tmp_path) -> None:
    metadata, geometry, evidence, ownership = _components()
    path = tmp_path / "v1"
    VoxelMapSnapshot.commit(path, metadata, geometry, evidence, ownership)
    restored = VoxelMapSnapshot.load(path)
    assert restored.metadata.schema_version == 1
    assert restored.registry is None
```

Add tests that schema v2 rejects a missing registry, ownership pointing to a missing entity, a corrupted `entities.jsonl`, and a semantic label outside the declared entity state.

- [ ] **Step 2: Run snapshot tests and verify schema rejection**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_snapshot.py
```

- [ ] **Step 3: Implement versioned snapshot files**

Allow metadata schema versions 1 and 2. Define:

```python
_DATA_FILES_V1 = ("metadata.json", "geometry.npz", "evidence.npz", "ownership.npz")
_DATA_FILES_V2 = (*_DATA_FILES_V1, "entities.jsonl")
```

Add `registry: EntityRegistry | None` to `VoxelMapSnapshot`. Extend `VoxelMapSnapshot.commit` with keyword-only `registry: EntityRegistry | None = None`; require a registry for schema v2 and forbid one for schema v1. Write and hash `entities.jsonl` atomically inside the temporary snapshot directory. `load` selects the exact file set from metadata schema, verifies all hashes before deserialization, and reconstructs the registry from the schema-v2 JSONL metadata record written in Task 8.

Make `Oviv2Runtime.commit` emit schema v2 and pass its registry. Update the evaluator to use `snapshot.registry` for v2. Keep `--entity-info` as an optional required fallback only when loading schema v1.

- [ ] **Step 4: Run snapshot, evaluator, and full OVIV2 tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2 tests/evaluation/test_evaluate_oviv2_replica_cli.py \
  tests/evaluation/test_oviv2_replica.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 10**

```bash
git add src/oviv2/snapshot.py src/oviv2/runtime.py \
  scripts/evaluation/evaluate_oviv2_replica.py tests/oviv2/test_snapshot.py \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py
git commit -m "feat: persist OVIV2 semantic entities in snapshots"
```

### Task 11: Wire Stage 1 Into The Replica Runner

**Files:**
- Modify: `scripts/run_oviv2_replica.py:222-405`
- Create: `configs/oviv2_replica_room0_precision_stage1.json`
- Modify: `tests/evaluation/test_run_oviv2_replica_cli.py`

- [ ] **Step 1: Write failing runner configuration tests**

```python
def test_precision_config_enables_feature_aware_owner_semantics() -> None:
    config = json.loads(Path("configs/oviv2_replica_room0_precision_stage1.json").read_text())
    runtime = _runtime_config(config)
    assert config["semantic_mode"] == "owner_authoritative"
    assert runtime.tracker.association.visual_weight > 0.0
    assert runtime.registry.association.semantic_weight > 0.0
    assert runtime.registry.prototype_top_k == 3
    assert runtime.registry.view_top_k == 10
```

Extend `_write_fixture` so its frontend manifest contains `"provenance_sha256": {"clip_model": "fixture-clip-sha256"}` and its cache payload contains two-dimensional `image_feats` and `text_feats`. Add a test that the feature model ID is `cached-clip:fixture-clip-sha256` and is recorded in the run manifest.

- [ ] **Step 2: Run the focused runner test and verify missing config fields**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_run_oviv2_replica_cli.py
```

Expected: fail because the precision config and association fields do not exist.

- [ ] **Step 3: Add isolated configuration and runner wiring**

Copy the current room0 geometry, structure, and evaluator values into the new JSON config. Add explicit Stage 1 values:

```json
{
  "semantic_mode": "owner_authoritative",
  "feature_mode": "cached_image",
  "association_min_directed_overlap": 0.10,
  "association_bounds_expansion_m": 0.10,
  "association_max_centroid_distance_m": 0.60,
  "association_minimum_score": 0.45,
  "association_geometry_weight": 0.20,
  "association_overlap_weight": 0.25,
  "association_visual_weight": 0.30,
  "association_semantic_weight": 0.20,
  "association_temporal_weight": 0.05,
  "semantic_conflict_confidence": 0.75,
  "semantic_conflict_visual_override": 0.80,
  "ambiguous_edge_score": 0.60,
  "third_view_min_score": 0.70,
  "prototype_top_k": 3,
  "prototype_merge_cosine": 0.90,
  "view_top_k": 10,
  "view_minimum_novelty_cosine": 0.10
}
```

Retain every existing path and static mapping parameter from `configs/oviv2_replica_room0.json`. In the runner, derive the cache feature model ID from `frontend_manifest["provenance_sha256"]["clip_model"]`, pass it to `CachedFrontendAdapter`, parse association/semantic-memory configs, include Stage 1 counters per frame, commit schema v2, and call `derive_labeled_mesh(snapshot.geometry, snapshot.evidence, snapshot.ownership, entity_semantics=runtime.registry.semantic_labels())`.

- [ ] **Step 4: Run runner/config tests and a two-frame smoke run**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2 tests/evaluation/test_evaluate_oviv2_replica_cli.py
```

Then run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/run_oviv2_replica.py \
  --config configs/oviv2_replica_room0_precision_stage1.json \
  --num-frames 2 \
  --output outputs/oviv2_room0_precision_stage1_2f \
  --skip-evaluation
```

Expected: exit 0; snapshot metadata schema is 2; `entities.jsonl` is present inside the snapshot; run manifest contains the feature-model hash and association configuration.

- [ ] **Step 5: Commit Task 11**

```bash
git add scripts/run_oviv2_replica.py configs/oviv2_replica_room0_precision_stage1.json \
  tests/evaluation/test_run_oviv2_replica_cli.py
git commit -m "feat: enable OVIV2 precision backend runner"
```

### Task 12: Gate The 20-Frame And 200-Frame Room0 Runs

**Files:**
- No source changes unless a test exposes a correctness defect.
- Generated immutable outputs: `outputs/oviv2_room0_precision_stage1_20f/` and `outputs/oviv2_room0_precision_stage1_200f/`.

- [ ] **Step 1: Run the 20-frame build and evaluation**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/run_oviv2_replica.py \
  --config configs/oviv2_replica_room0_precision_stage1.json \
  --num-frames 20 \
  --output outputs/oviv2_room0_precision_stage1_20f
```

Expected: exit 0, no non-finite metrics, more than one predicted entity, and no snapshot checksum errors.

- [ ] **Step 2: Inspect the 20-frame association counters**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -c \
'import json; p=json.load(open("outputs/oviv2_room0_precision_stage1_20f/timing.json")); f=p["frames"]; print({"entities": sum(x["new_entity_count"] for x in f), "matches": sum(x["matched_entity_count"] for x in f), "conflicts": sum(x["association_conflict_count"] for x in f)})'
```

Expected: all counters are non-negative, at least one match exists, and the final entity count is neither zero nor equal to every observation count.

- [ ] **Step 3: Run the 200-frame room0 build and evaluation only after the 20-frame gate passes**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/run_oviv2_replica.py \
  --config configs/oviv2_replica_room0_precision_stage1.json \
  --num-frames 200 \
  --output outputs/oviv2_room0_precision_stage1_200f
```

Expected: exit 0 and immutable evaluation artifacts are present.

- [ ] **Step 4: Compare Stage 1 with the verified baseline gate**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -c \
'import json; b=json.load(open("outputs/oviv2_replica8_frozen/room0/evaluation/metrics.json")); n=json.load(open("outputs/oviv2_room0_precision_stage1_200f/evaluation/metrics.json")); print({"baseline": {k:b[k] for k in ("miou","ap50","f5")}, "stage1": {k:n[k] for k in ("miou","ap50","f5")}, "delta_miou": n["miou"]-b["miou"], "delta_ap50": n["ap50"]-b["ap50"]})'
```

Acceptance:

- room0 F@5cm is at least `0.90`;
- room0 AP50 is not below `0.0417`;
- Stage 1 is retained only if mIoU improves, or AP50 improves while mIoU drops by no more than `0.01`;
- the output entity count and over-merge diagnostic improve for metric-backed reasons; entity count alone is not a gate.

- [ ] **Step 5: Run Stage 0 diagnostics on the Stage 1 aligned arrays**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/diagnose_oviv2_room0.py \
  --pred-semantic outputs/oviv2_room0_precision_stage1_200f/evaluation/gt_aligned_semantic_ids.npy \
  --pred-entity outputs/oviv2_room0_precision_stage1_200f/evaluation/gt_aligned_instance_ids.npy \
  --gt-mesh /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --manifest configs/evaluation/manifests/replica8.json \
  --scene room0 \
  --output outputs/oviv2_room0_precision_stage1_200f/diagnostics.json
```

Expected: `headline_eligible=false`; use the oracle gap and over-merge/fragmentation counts to decide whether Stage 2 should prioritize dense semantics or further association work.

### Task 13: Final Regression Verification

**Files:**
- No production edits expected.

- [ ] **Step 1: Run formatting and whitespace checks**

```bash
git diff --check 2f24b6d..HEAD
```

Expected: no output.

- [ ] **Step 2: Run the complete OVIV2 and evaluator suites**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2 \
  tests/evaluation/test_oviv2_replica.py \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py \
  tests/evaluation/test_oviv2_diagnostics.py \
  tests/evaluation/test_diagnose_oviv2_room0.py
```

Expected: all pass.

- [ ] **Step 3: Verify the final commit scope**

```bash
git status --short
git log --oneline --decorate -15
```

Expected: implementation commits contain only the files named in this plan; pre-existing dirty legacy files remain uncommitted and unchanged by this work.

- [ ] **Step 4: Record exact metrics for the Stage 2 planning decision**

Print the Stage 1 metrics and diagnostics:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -c \
'import json; m=json.load(open("outputs/oviv2_room0_precision_stage1_200f/evaluation/metrics.json")); d=json.load(open("outputs/oviv2_room0_precision_stage1_200f/diagnostics.json")); print(json.dumps({"metrics": {k:m[k] for k in ("miou","macc","f_miou","ap25","ap50","f5")}, "diagnostics": d["entity_geometry"], "oracle_miou": d["semantic"]["oracle_miou"]}, indent=2, sort_keys=True))'
```

Expected: finite metrics with the acceptance decision stated from Task 12. Do not update the benchmark table from a room0 development run.
