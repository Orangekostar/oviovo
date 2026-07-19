# Table 1 Baseline Credibility Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct OVI-MAP instance coverage, produce separated neutral and official-style diagnostics, and establish a canonical ConceptGraphs room0 gate before replacing any Table 1 result.

**Architecture:** OVI-MAP mesh instances and semantic-feature instances are loaded as two related populations and joined by authoritative global instance ID. Neutral metrics remain the only source of headline tokens; released evaluator output is captured in a diagnostic schema. ConceptGraphs canonical execution uses the released SAM/Grounded-SAM configuration in an external build directory and a tracked gate validator.

**Tech Stack:** Python 3.10, NumPy, SciPy, plyfile, PyTorch, transformers, pytest, existing neutral baseline contracts, external Conda/Docker baseline environments.

---

### Task 1: Preserve Featureless OVI-MAP Mesh Instances

**Files:**
- Modify: `tests/evaluation/test_baseline_adapters.py`
- Modify: `src/evaluation/baselines/adapters.py`

- [ ] **Step 1: Write the failing adapter test**

Add:

```python
def test_ovimap_adapter_keeps_featureless_mesh_instance() -> None:
    artifact = adapt_ovimap(
        {7: {"color": np.array([10, 20, 30], dtype=np.uint8)}},
        points_by_color={
            (10, 20, 30): np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        },
        scene_id="room0",
        timestamp=10.0,
        upstream_commit="58a804e",
        runtime=_runtime(),
    )

    entity = artifact.snapshot.entities[0]
    assert entity.entity_id == "ovimap:7"
    assert entity.semantic_embedding is None
    assert entity.semantic_label is None
    assert entity.metadata["observation_count"] == 0
    np.testing.assert_allclose(entity.points_xyz, [[0.0, 0.0, 1.0]])
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_baseline_adapters.py::test_ovimap_adapter_keeps_featureless_mesh_instance -q
```

Expected: FAIL with missing `feat` in `adapt_ovimap`.

- [ ] **Step 3: Implement optional semantic features**

In `adapt_ovimap`, replace unconditional feature loading with:

```python
        raw_features = instance.get("feat")
        embedding = None
        if raw_features is not None:
            features = np.asarray(raw_features, dtype=np.float32)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if features.ndim != 2 or not len(features):
                raise ValueError("OVI-MAP semantic features must be a non-empty 2D array")
            visibility = np.asarray(instance.get("vis_area", ()), dtype=np.float32).reshape(-1)
            if len(visibility) == len(features) and len(visibility):
                features = features[-8:]
                weights = visibility[-8:]
                weights = weights / (float(weights.sum()) + 1e-6)
                embedding = (features * weights[:, None]).sum(axis=0)
            else:
                embedding = features.mean(axis=0)
```

Keep `frame_id` optional and set `observation_count=len(frames)`.

- [ ] **Step 4: Run focused adapter tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_baseline_adapters.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1 only**

```bash
git add src/evaluation/baselines/adapters.py tests/evaluation/test_baseline_adapters.py
git commit -m "fix: preserve featureless OVI-MAP instances"
```

---

### Task 2: Bind the Full OVI-MAP Instance Mesh

**Files:**
- Modify: `tests/evaluation/test_ovimap_static_loader.py`
- Modify: `src/evaluation/baselines/ovimap.py`
- Modify: `scripts/evaluation/evaluate_replica_static.py`

- [ ] **Step 1: Write failing join tests**

Import `bind_mesh_instances` and add:

```python
def test_bind_mesh_instances_keeps_logged_instances_without_features() -> None:
    features = {
        7: {"feat": [[1.0, 0.0]], "frame_id": [0], "color": [7, 7, 7]}
    }
    colors = {7: (10, 20, 30), 8: (40, 50, 60)}
    points = {
        (10, 20, 30): np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        (40, 50, 60): np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    }

    bound = bind_mesh_instances(features, colors, points)

    assert set(bound) == {7, 8}
    assert bound[7]["color"] == (10, 20, 30)
    assert bound[8] == {"color": (40, 50, 60)}


def test_bind_mesh_instances_rejects_feature_id_missing_from_log() -> None:
    with pytest.raises(ValueError, match="missing from instance color log"):
        bind_mesh_instances(
            {9: {"feat": [[1.0]]}},
            {7: (10, 20, 30)},
            {(10, 20, 30): np.zeros((1, 3), dtype=np.float32)},
        )
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_static_loader.py -q
```

Expected: collection failure because `bind_mesh_instances` does not exist.

- [ ] **Step 3: Implement the join**

Add to `src/evaluation/baselines/ovimap.py`:

```python
def bind_mesh_instances(
    semantic_instances: Mapping[int, Mapping[str, Any]],
    colors_by_instance: Mapping[int, tuple[int, int, int]],
    points_by_color: Mapping[tuple[int, int, int], np.ndarray],
) -> dict[int, dict[str, Any]]:
    feature_ids = {int(value) for value in semantic_instances}
    missing = sorted(feature_ids - {int(value) for value in colors_by_instance})
    if missing:
        raise ValueError(f"OVI-MAP feature instance IDs missing from instance color log: {missing}")
    bound: dict[int, dict[str, Any]] = {}
    for instance_id, color in sorted(colors_by_instance.items()):
        normalized_color = tuple(int(value) for value in color)
        if normalized_color not in points_by_color:
            continue
        record = dict(semantic_instances.get(int(instance_id), {}))
        record["color"] = normalized_color
        bound[int(instance_id)] = record
    return bound
```

Add required `Any` and `Mapping` imports.

- [ ] **Step 4: Load all logged colors before joining**

In `_load_ovimap`:

```python
    with args.instances_file.open("rb") as handle:
        semantic_instances = pickle.load(handle)
    colors_by_instance = parse_instance_color_log(args.instance_color_log)
    points_by_color, background_xyz = load_instance_mesh(
        args.instance_mesh,
        colors_by_instance.values(),
    )
    instances = bind_mesh_instances(
        semantic_instances,
        colors_by_instance,
        points_by_color,
    )
```

Require `--instance-color-log` for OVI-MAP formal evaluation and remove the semantic-subset-only requested-color path.

- [ ] **Step 5: Run loader and CLI regressions**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_static_loader.py \
  tests/evaluation/test_evaluate_replica_static_cli.py \
  tests/evaluation/test_baseline_adapters.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2 only**

```bash
git add src/evaluation/baselines/ovimap.py scripts/evaluation/evaluate_replica_static.py \
  tests/evaluation/test_ovimap_static_loader.py
git commit -m "fix: evaluate full OVI-MAP instance meshes"
```

---

### Task 3: Prove Class-Agnostic AP Is Independent of Semantic Features

**Files:**
- Modify: `tests/evaluation/test_baseline_static_metrics.py`

- [ ] **Step 1: Add the regression fixture**

Create a `MapSnapshot` with two mesh-backed entities: one with a semantic label and one with `semantic_label=None`. Create matching GT instances and call `evaluate_static_snapshot` with `instance_vocabulary` set.

Assert:

```python
assert result["instance"]["predicted_instance_count"] == 2
assert result["instance"]["ap25"] == pytest.approx(1.0)
assert result["semantic"]["matched_point_ratio"] == pytest.approx(0.5)
```

- [ ] **Step 2: Verify the test fails against the semantic-subset loader fixture**

Run the focused test before applying Task 2's loader implementation when replaying TDD. Expected: only one predicted instance.

- [ ] **Step 3: Run after Tasks 1-2 and verify GREEN**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_baseline_static_metrics.py -q
```

Expected: PASS with two predicted instances.

- [ ] **Step 4: Commit the regression**

```bash
git add tests/evaluation/test_baseline_static_metrics.py
git commit -m "test: cover featureless OVI-MAP instance AP"
```

---

### Task 4: Capture Released OVI-MAP Diagnostics Without Renaming Metrics

**Files:**
- Create: `src/evaluation/baselines/ovimap_diagnostics.py`
- Create: `scripts/evaluation/capture_ovimap_diagnostics.py`
- Create: `tests/evaluation/test_ovimap_diagnostics.py`

- [ ] **Step 1: Write parser tests**

Test released instance output:

```python
INSTANCE_OUTPUT = """mIoU\twIoU\tmP@75\tmR@75\tmP@50\tmR@50\tmP@25\tmR@25
0.363\t0.500\t0.220\t0.180\t0.508\t0.410\t0.767\t0.620
"""

def test_parser_preserves_released_metric_names() -> None:
    payload = parse_official_instance_output(INSTANCE_OUTPUT, scene_id="room0")
    assert payload["metrics"]["mean_precision_at_25"] == pytest.approx(0.767)
    assert payload["metrics"]["mean_recall_at_25"] == pytest.approx(0.620)
    assert "ap25" not in payload["metrics"]
```

Also test malformed headers, non-finite values, and extra/missing rows.

- [ ] **Step 2: Run and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_diagnostics.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement strict TSV parsing**

Implement `parse_official_instance_output(text, scene_id)` and `parse_official_semantic_output(text, scene_id)`. Return sorted JSON-compatible dictionaries with `schema_version`, `scene_id`, `source_protocol="released_ovimap"`, exact source headers, normalized descriptive metric keys, and no token bindings.

- [ ] **Step 4: Implement the capture CLI**

CLI arguments:

```text
--scene-id
--instance-stdout
--semantic-stdout
--neutral-metrics
--output
```

Hash all three inputs and atomically write `official_style_diagnostics.json`.

- [ ] **Step 5: Run focused tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/evaluation/baselines/ovimap_diagnostics.py \
  scripts/evaluation/capture_ovimap_diagnostics.py \
  tests/evaluation/test_ovimap_diagnostics.py
git commit -m "feat: capture OVI-MAP protocol diagnostics"
```

---

### Task 5: Re-Evaluate OVI-MAP From Existing Replica Outputs

**Files:**
- Create externally: `/home/ww/oviovo_baseline_runs/20260714_non_oviovo_baselines/ovimap/common_eval_full_instances/`
- Create externally: `/home/ww/oviovo_baseline_runs/20260714_non_oviovo_baselines/ovimap/result_provenance_full_instances.json`
- Create: `docs/paper/results/baselines/ovimap/replica/20260719-s10-200f-full-instances-v2/result.json`

- [ ] **Step 1: Run room0 neutral evaluation**

Copy the existing `run_common_eval_replica8.sh` to an external versioned wrapper using `common_eval_full_instances` as output. Do not edit the old output directory.

Run room0 with the existing repaired artifacts and verify:

```bash
jq '{semantic:.metrics.semantic,instance:.metrics.instance,geometry:.metrics.geometry}' \
  /home/ww/oviovo_baseline_runs/20260714_non_oviovo_baselines/ovimap/common_eval_full_instances/room0/metrics.json
```

Required gate: `predicted_instance_count` is greater than the old value `9`, geometry F5 remains finite, and all metrics are finite.

- [ ] **Step 2: Run released room0 evaluators as diagnostics**

Use the external OVI-MAP build/container to generate the released `instance_map_gt_200.ply` and `semantic_map_gt_200.ply`, then run the unmodified reference `eval_inst_seg.py` and `eval_sem_seg.py`. Save stdout, commands, exit codes, and hashes under `common_eval_full_instances/room0/official_diagnostics/`.

- [ ] **Step 3: Capture the room0 diagnostic JSON**

Run `capture_ovimap_diagnostics.py` and verify it has no `token_bindings` or headline aliases.

- [ ] **Step 4: Run all eight neutral evaluations**

Run each scene into its own directory. Aggregate only after all eight status records have `exit_status=0`.

- [ ] **Step 5: Finalize a new result**

Use a new run ID containing `full-instances`. Preserve the old result. Add all corrected metrics, diagnostics, runtime, commands, deviations, and hashes to provenance before calling `finalize_static_result.py`.

- [ ] **Step 6: Import only after hash audit**

Verify every result path/SHA-256 pair, then run `tools/import_benchmark_results.py`. Confirm canonical template hashes and all OVIOVO/OVIV2 token ownership rules.

- [ ] **Step 7: Commit the new result and derived registry files**

Stage only the new result, registry/derived tables if changed, and scoped report updates.

---

### Task 6: Define and Validate the ConceptGraphs Canonical Gate

**Files:**
- Create: `configs/evaluation/baselines/conceptgraphs_canonical_replica.json`
- Create: `scripts/evaluation/validate_conceptgraphs_gate.py`
- Create: `tests/evaluation/test_validate_conceptgraphs_gate.py`

- [ ] **Step 1: Write a failing configuration identity test**

Assert the configuration declares:

```json
{
  "frontend": "sam_segment_all",
  "gsa_variant": "none",
  "class_agnostic": true,
  "mask_conf_threshold": 0.95,
  "sim_threshold": 1.2,
  "dbscan_eps": 0.1,
  "merge_interval": 20,
  "merge_visual_sim_thresh": 0.8,
  "merge_text_sim_thresh": 0.8,
  "headline_stride": 10,
  "official_diagnostic_stride": 5
}
```

The test rejects `yolo_world`, `mobile_sam`, GT-only class suppression, or `n_exclude=6` in headline mode.

- [ ] **Step 2: Run and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_validate_conceptgraphs_gate.py -q
```

Expected: missing config/validator failure.

- [ ] **Step 3: Implement the config and validator**

The validator reads a status JSON plus detection/map artifacts and requires:

- exact upstream and Grounded-SAM commits;
- `exit_status=0` for frontend and mapper;
- positive mask count and mapped object count;
- non-empty map file;
- finite neutral matched-point ratio;
- all expected weight and config hashes.

It writes a small gate JSON atomically and returns non-zero on failure.

- [ ] **Step 4: Run validator tests**

Expected: PASS for a complete synthetic fixture and explicit failures for empty masks, wrong frontend identity, GT suppression, and missing hashes.

- [ ] **Step 5: Commit Task 6**

```bash
git add configs/evaluation/baselines/conceptgraphs_canonical_replica.json \
  scripts/evaluation/validate_conceptgraphs_gate.py \
  tests/evaluation/test_validate_conceptgraphs_gate.py
git commit -m "feat: gate canonical ConceptGraphs evaluation"
```

---

### Task 7: Execute Canonical ConceptGraphs room0

**Files:**
- Create externally: `/home/ww/oviovo_baseline_builds/conceptgraphs-canonical/`
- Create externally: `/home/ww/oviovo_baseline_runs/20260719_conceptgraphs_canonical/room0/`

- [ ] **Step 1: Freeze local assets**

Record SHA-256 for the existing RAM and GroundingDINO weights. Locate or download the official SAM ViT-H checkpoint to `/home/ww/oviovo_benchmark_assets/weights/` and record URL, license, hash, and date.

- [ ] **Step 2: Freeze external source commits**

Use the existing `/home/ww/vv/lifelongmap_benchmack/concept-graphs/Grounded-Segment-Anything` checkout only as a source reference. Create a clean external build copy at the recorded commit; do not edit the reference checkout.

- [ ] **Step 3: Run canonical room0 frontend**

Use `generate_gsa_results.py --class_set none --stride 10` with SAM segment-all. Capture `/usr/bin/time -v`, GPU dmon, command, environment freeze, config hash, weight hashes, and output mask count.

- [ ] **Step 4: Run released object mapping**

Use the exact canonical thresholds from Task 6 and write to an isolated room0 output directory.

- [ ] **Step 5: Run neutral room0 evaluation and gate**

Run the neutral evaluator with the frozen Replica-41 vocabulary, then `validate_conceptgraphs_gate.py`.

- [ ] **Step 6: Decide the long-run gate mechanically**

Start the remaining seven scenes only if gate status is `VERIFIED`. Otherwise record `BLOCKED` with the real failing command, exit code, log, missing dependency/artifact, and next action.

---

## Completion Gate

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q -p no:cacheprovider \
  tests/evaluation/test_baseline_adapters.py \
  tests/evaluation/test_ovimap_static_loader.py \
  tests/evaluation/test_baseline_static_metrics.py \
  tests/evaluation/test_evaluate_replica_static_cli.py \
  tests/evaluation/test_ovimap_diagnostics.py \
  tests/evaluation/test_validate_conceptgraphs_gate.py \
  tests/evaluation/test_import_benchmark_results.py \
  tests/test_benchmark_table_package.py
```

Then verify all result hashes and run:

```bash
cd /home/ww/oviovo_aaai_workspace
VERIFY_ENVIRONMENTS=0 bash scripts/verify_workspace.sh
```

Success requires a corrected OVI-MAP result from a new run directory, a room0 canonical ConceptGraphs gate outcome, unchanged reference checkouts, and no changes to pre-existing dirty legacy pipeline files.
