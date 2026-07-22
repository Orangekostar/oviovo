# OVIV2 TESSE-CD v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, freeze, run twice, and strictly finalize the causal OVIV2 Stage3 TESSE-CD evaluation needed to replace the paper's T2 placeholders with real results.

**Architecture:** Keep the Stage3 online runtime at lineage commit `47962fbd9f363c0696cc5016f8ab42f83a3bf7e5`, add a validated TESSE-CD RGB-D adapter, frame-local object and RADSeg caches, immutable causal checkpoints, and a deterministic snapshot-to-neutral-artifact exporter. Port the reviewed common-v2 protocol as a narrow source closure, freeze Apartment-selected lifecycle parameters before held-out Office evaluation, then run both scenes twice and bind all results and evidence by hash.

**Tech Stack:** Python 3.10/3.12, NumPy, SciPy, Pillow, PyYAML, Open3D, pytest, YOLO-World, MobileSAM, CLIP, RADSeg/RADIO, Git worktrees, canonical JSON and SHA-256 manifests.

---

## Execution Graph And Ownership

Use one clean integration worktree at `/home/ww/.config/superpowers/worktrees/tessecd-stage3-v1`. The filesystem path intentionally excludes `oviv2`, `baseline`, and `prediction` because the reviewed TESSE target-source guards reject those path components. Create branch `feature/oviv2-tessecd-v1` from the commit containing this plan. Every parallel lane uses its own worktree and branch (`tessecd-lane-protocol`, `tessecd-lane-dataset`, `tessecd-lane-naming`, `tessecd-lane-runtime`, then wave-specific lane worktrees) so simultaneous commits cannot stage another worker's files. The integration owner cherry-picks completed lane commits in dependency order.

After Task 1, dispatch these non-overlapping lanes:

| Wave | Lane | Tasks | Exclusive paths |
|---|---|---|---|
| A | Protocol | 2, 3 | `configs/evaluation/manifests/tesse_cd*`, `configs/evaluation/semantic_aliases/`, common-v2 scripts/modules/tests, TESSE finalizers |
| A | Dataset | 4 | `src/datasets/tesse_cd.py`, TESSE vocabulary/config files, dataset/config tests |
| A | Naming | 5 | `tools/benchmark_tables.py`, `tools/import_benchmark_results.py`, active paper token/table files, their tests |
| A | Runtime foundation | 6 | `src/oviv2/runner_config.py`, `src/oviv2/{snapshot,runtime}.py`, focused Stage3 tests |
| B | Object cache | 7 | TESSE frontend orchestrator and its tests |
| B | Dense cache | 8 | dense precompute TESSE support and its tests |
| B | Neutral export | 9 | `src/evaluation/oviv2_tesse.py` and exporter tests |
| B | Occlusion target | 11 steps 1-5 | occlusion target derivation/module/tests only |
| C | Online runner | 10 | TESSE runner, scene configs, runner tests |
| C | Occlusion runtime | 11 steps 6-10 | runtime policy, occlusion evaluator, focused tests |
| C | Official bridge | 12 | neutral-to-Khronos bridge, official evaluator wrapper, summarizer, compatibility source/tests |

Do not let two workers edit `benchmark_tokens.tsv`, generated table files, either TESSE finalizer, `src/oviv2/runtime.py`, or `tests/oviv2/test_runtime.py`. Create every wave-B/C lane from the integration commit after its dependency wave passes. Remove lane worktrees only after their commits are cherry-picked and verified. Merge and run the stated gate after every wave. Tasks 13-17 are integration/experiment gates and are serial except where explicit GPU or repeat parallelism is stated.

### Task 1: Create The Clean Integration Worktree

**Files:**
- Verify: `docs/superpowers/specs/2026-07-22-oviv2-tessecd-v1-design.md`
- Verify: `docs/superpowers/plans/2026-07-22-oviv2-tessecd-v1.md`

- [ ] **Step 1: Create the isolated worktree with the required skill**

Run from `/home/ww/.config/superpowers/worktrees/oviovo/oviv2-stage3-final-batch`:

```bash
git worktree add -b feature/oviv2-tessecd-v1 \
  /home/ww/.config/superpowers/worktrees/tessecd-stage3-v1 HEAD
```

Expected: a clean worktree whose branch is `feature/oviv2-tessecd-v1` and whose path contains none of the forbidden target-source markers.

- [ ] **Step 2: Verify lineage and cleanliness**

Run:

```bash
cd /home/ww/.config/superpowers/worktrees/tessecd-stage3-v1
git merge-base --is-ancestor 47962fbd9f363c0696cc5016f8ab42f83a3bf7e5 HEAD
git status --short
```

Expected: the ancestry command exits 0 and `git status --short` emits no lines.

- [ ] **Step 3: Create isolated wave-A lane worktrees**

Run:

```bash
git worktree add -b feature/oviv2-tessecd-v1-protocol \
  /home/ww/.config/superpowers/worktrees/tessecd-lane-protocol HEAD
git worktree add -b feature/oviv2-tessecd-v1-dataset \
  /home/ww/.config/superpowers/worktrees/tessecd-lane-dataset HEAD
git worktree add -b feature/oviv2-tessecd-v1-naming \
  /home/ww/.config/superpowers/worktrees/tessecd-lane-naming HEAD
git worktree add -b feature/oviv2-tessecd-v1-runtime \
  /home/ww/.config/superpowers/worktrees/tessecd-lane-runtime HEAD
```

Expected: four clean lane worktrees. Each worker commits only in its assigned lane; no worker edits the integration worktree directly.

- [ ] **Step 4: Run the Stage3 baseline gate**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q \
  tests/oviv2/test_runtime.py \
  tests/oviv2/test_snapshot.py \
  tests/oviv2/test_visibility.py \
  tests/oviv2/test_ownership.py \
  tests/evaluation/test_run_oviv2_replica_cli.py
```

Expected: all selected tests pass. Record any environment-only skips; do not change code to hide a real failure.

### Task 2: Port The Reviewed TESSE-CD Protocol Closure

**Files:**
- Modify: `src/evaluation/__init__.py`
- Create: `src/evaluation/baselines/dynamic_metrics.py`
- Create: `src/evaluation/baselines/tesse_semantics.py`
- Create: `configs/evaluation/manifests/tesse_cd.json`
- Create: `scripts/evaluation/derive_tesse_cd_causal_schedule.py`
- Create: `configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json`
- Create: `scripts/evaluation/export_tesse_cd_rgbd.py`
- Create: `scripts/evaluation/derive_tesse_cd_common_v2.py`
- Create: `configs/evaluation/manifests/tesse_cd_common_v2.json`
- Create: `configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml`
- Create: `scripts/evaluation/export_tesse_temporal_artifact.py`
- Create: `scripts/evaluation/evaluate_tesse_cd_common_v2.py`
- Test: `tests/evaluation/test_evaluation_package_imports.py`
- Test: `tests/evaluation/test_tesse_cd_manifest.py`
- Test: `tests/evaluation/test_derive_tesse_cd_causal_schedule.py`
- Test: `tests/evaluation/test_export_tesse_cd_rgbd.py`
- Test: `tests/evaluation/test_dynamic_baseline_metrics.py`
- Test: `tests/evaluation/test_dynamic_metrics_common_v2.py`
- Test: `tests/evaluation/test_derive_tesse_cd_common_v2.py`
- Test: `tests/evaluation/test_tesse_semantics.py`
- Test: `tests/evaluation/test_export_tesse_temporal_artifact.py`
- Test: `tests/evaluation/test_evaluate_tesse_cd_common_v2.py`

- [ ] **Step 1: Add failing import and protocol identity tests**

Add tests that import `src.evaluation.baselines.dynamic_metrics` without eagerly importing `src.oviv2`, validate the two canonical scenes, and reject source/schedule/target hash drift. The import assertion is:

```python
def test_dynamic_metrics_import_does_not_load_oviv2() -> None:
    sys.modules.pop("src.oviv2", None)
    importlib.import_module("src.evaluation.baselines.dynamic_metrics")
    assert "src.oviv2" not in sys.modules
```

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q \
  tests/evaluation/test_evaluation_package_imports.py \
  tests/evaluation/test_tesse_cd_manifest.py
```

Expected: collection or import fails because the TESSE modules and manifests are absent.

- [ ] **Step 2: Port only the reviewed source closure**

Use the reviewed files in `/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates` as the source, but apply them as explicit patches to the target worktree. Do not copy that worktree's tracked changes, baseline runners, compatibility patches, results, or generated outputs. Change `src/evaluation/__init__.py` to lazy-load the existing Replica API:

```python
def __getattr__(name: str) -> object:
    if name in {"ReplicaGroundTruth", "evaluate_replica_snapshot"}:
        from . import oviv2_replica

        return getattr(oviv2_replica, name)
    raise AttributeError(name)
```

Keep the existing `src/evaluation/contracts.py` and `src/evaluation/exporters/oviovo.py`; the port must use those Stage3-compatible versions.

- [ ] **Step 3: Replace substring path guards with role-aware path checks**

In both target derivation and common-v2 finalization, validate explicit source roles and reject forbidden *path components* only for declared prediction inputs:

```python
FORBIDDEN_PREDICTION_ROLES = {
    "method_output", "method_outputs", "prediction", "predictions"
}

def reject_prediction_source(record: Mapping[str, object]) -> None:
    role = str(record.get("role", "")).strip().lower()
    if role in FORBIDDEN_PREDICTION_ROLES:
        raise ValueError("ground-truth target source declares a prediction role")
```

Add tests proving that a clean source file below a directory named `tessecd-stage3-v1` is accepted, while a record whose explicit role is `prediction` is rejected even if its path looks harmless.

- [ ] **Step 4: Regenerate the checked schedule and common-v2 contract**

Do not copy the checked JSON byte-for-byte because it binds the source worktree path. Run the ported generator against the canonical manifest, then build the common contract from the regenerated schedule:

```bash
python scripts/evaluation/derive_tesse_cd_causal_schedule.py \
  --manifest configs/evaluation/manifests/tesse_cd.json \
  --output configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json

python scripts/evaluation/derive_tesse_cd_common_v2.py contract \
  --manifest configs/evaluation/manifests/tesse_cd.json \
  --schedule configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json \
  --aliases configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml \
  --output configs/evaluation/manifests/tesse_cd_common_v2.json
```

Expected: Apartment and Office schedules are regenerated from hash-bound inputs. In-repository source paths are serialized relative to the repository root and resolved against that root, while canonical external assets remain absolute/hash-bound. A relocation test copies the repository fixture to a different temporary root and obtains the same contract hash.

- [ ] **Step 5: Run the protocol gate**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q \
  tests/evaluation/test_evaluation_package_imports.py \
  tests/evaluation/test_tesse_cd_manifest.py \
  tests/evaluation/test_derive_tesse_cd_causal_schedule.py \
  tests/evaluation/test_export_tesse_cd_rgbd.py \
  tests/evaluation/test_dynamic_baseline_metrics.py \
  tests/evaluation/test_dynamic_metrics_common_v2.py \
  tests/evaluation/test_derive_tesse_cd_common_v2.py \
  tests/evaluation/test_tesse_semantics.py \
  tests/evaluation/test_export_tesse_temporal_artifact.py \
  tests/evaluation/test_evaluate_tesse_cd_common_v2.py
```

Expected: all tests pass, including checked-asset binding tests.

- [ ] **Step 6: Commit the protocol closure**

```bash
git add src/evaluation configs/evaluation scripts/evaluation/derive_tesse_cd_causal_schedule.py \
  scripts/evaluation/export_tesse_cd_rgbd.py \
  scripts/evaluation/derive_tesse_cd_common_v2.py \
  scripts/evaluation/export_tesse_temporal_artifact.py \
  scripts/evaluation/evaluate_tesse_cd_common_v2.py tests/evaluation
git commit -m "feat: port TESSE-CD common-v2 protocol"
```

### Task 3: Add Strict OVIV2 TESSE Finalizers

**Files:**
- Create: `scripts/evaluation/finalize_tesse_common_v2.py`
- Create: `scripts/evaluation/finalize_tesse_t2.py`
- Modify: `scripts/evaluation/evaluate_tesse_cd_common_v2.py`
- Test: `tests/evaluation/test_finalize_tesse_common_v2.py`
- Test: `tests/evaluation/test_finalize_tesse_t2.py`
- Test: `tests/evaluation/test_evaluate_tesse_cd_common_v2.py`

- [ ] **Step 1: Write failing OVIV2 registration tests**

Require the evaluator to accept a snapshot whose index and snapshot method are exactly `OVIV2`, and require both finalizers to reject `OVIOVO`. The expected registrations are:

```python
assert METHODS["OVIV2"] == {
    "display_label": "OVIV2",
    "mode": "online",
    "summary_mode": "causal_checkpoints",
    "eligible_for_ranking": True,
}
assert METHOD_MODES["OVIV2"] == "causal_checkpoints"
assert TABLE_MODES["OVIV2"] == "online"
```

Run:

```bash
pytest -q \
  tests/evaluation/test_evaluate_tesse_cd_common_v2.py \
  tests/evaluation/test_finalize_tesse_common_v2.py \
  tests/evaluation/test_finalize_tesse_t2.py
```

Expected: failures report unsupported method `OVIV2`.

- [ ] **Step 2: Register the exact causal and display modes**

Add:

```python
CAUSAL_SNAPSHOT_METHOD_LABELS["OVIV2"] = {"OVIV2"}

METHODS["OVIV2"] = {
    "display_label": "OVIV2",
    "mode": "online",
    "summary_mode": "causal_checkpoints",
    "eligible_for_ranking": True,
}

METHOD_MODES["OVIV2"] = "causal_checkpoints"
TABLE_MODES["OVIV2"] = "online"
DISPLAY_LABELS["OVIV2"] = "OVIV2"
```

The official finalizer must emit six `T2_OVIV2_{SCENE}_{METRIC}` bindings or source-bound unavailable bindings. The common finalizer must emit the four finite macro bindings. Both finalizers must bind primary and repeat input hashes.

- [ ] **Step 3: Validate inclusive and exclusive causal boundaries**

For checkpoint frame `t`, require both representations and their consistency:

```python
if checkpoint.get("consumed_through_frame") != frame_index:
    raise ValueError("checkpoint consumed-through frame mismatch")
if checkpoint.get("consumed_through_frame_exclusive") != frame_index + 1:
    raise ValueError("checkpoint exclusive causal boundary mismatch")
```

Add failure cases for `t + 1` in the inclusive field and `t` in the exclusive field.

- [ ] **Step 4: Run finalizer tests and commit**

Run:

```bash
pytest -q \
  tests/evaluation/test_evaluate_tesse_cd_common_v2.py \
  tests/evaluation/test_finalize_tesse_common_v2.py \
  tests/evaluation/test_finalize_tesse_t2.py
git add scripts/evaluation/evaluate_tesse_cd_common_v2.py \
  scripts/evaluation/finalize_tesse_common_v2.py \
  scripts/evaluation/finalize_tesse_t2.py tests/evaluation
git commit -m "feat: finalize causal OVIV2 TESSE results"
```

Expected: all selected tests pass and the commit contains only evaluator/finalizer files.

### Task 4: Add The TESSE RGB-D Adapter And Frozen Vocabularies

**Files:**
- Create: `src/datasets/tesse_cd.py`
- Create: `configs/evaluation/manifests/oviv2_tesse_cd_cache.json`
- Create: `configs/evaluation/vocabularies/tesse_cd_apartment.json`
- Create: `configs/evaluation/vocabularies/tesse_cd_apartment.txt`
- Create: `configs/evaluation/vocabularies/tesse_cd_office.json`
- Create: `configs/evaluation/vocabularies/tesse_cd_office.txt`
- Test: `tests/datasets/test_tesse_cd.py`
- Test: `tests/evaluation/test_oviv2_tesse_cache_config.py`

- [ ] **Step 1: Write failing adapter tests**

Create a five-frame fixture and require the API:

```python
dataset = TesseCdRgbdDataset(
    fixture_root,
    scene="apartment",
    export_manifest=fixture_root / "export_manifest.json",
    schedule_manifest=fixture_root / "schedule.json",
)
frame = dataset[2]
assert frame.frame_id == 2
assert frame.source_frame_id == 2
assert frame.timestamp == dataset.timestamp_ns(2) / 1_000_000_000
assert frame.depth.dtype == np.float32
```

Also reject an unknown scene, non-increasing timestamps, wrong RGB size, non-uint16 source depth, non-finite pose, mismatched camera intrinsics, missing frame, and changed export/schedule hash.

Run: `pytest -q tests/datasets/test_tesse_cd.py`

Expected: import fails because `src.datasets.tesse_cd` does not exist.

- [ ] **Step 2: Implement the immutable dataset records**

Add:

```python
@dataclass(frozen=True)
class TesseCdFrameRecord:
    frame_index: int
    timestamp_ns: int
    relative_timestamp_ns: int
    rgb_path: Path
    depth_path: Path
    camera_to_world: np.ndarray

class TesseCdRgbdDataset:
    EXPECTED_FRAMES = {"apartment": 1745, "office": 4346}
    WIDTH = 720
    HEIGHT = 480
    DEPTH_SCALE = 1000.0

    def __len__(self) -> int:
        return len(self.records)

    def timestamp_ns(self, index: int) -> int:
        return self.records[index].timestamp_ns
```

Load RGB and depth lazily in `__getitem__`; validate filenames, timestamps, poses, camera values, and manifest hashes in the constructor. Keep source frame index equal to dataset/cache index and convert millimeters to meters only when creating `Frame`.

- [ ] **Step 3: Freeze ordered scene vocabularies**

Write the official object lists in semantic-ID order:

```json
{"scene":"apartment","classes":["Fridge","Books","Chair","Vase","Couch","Drawer","Objects","Table","Bin","Humans"],"object_semantic_ids":[1,2,5,6,7,9,10,16,18,20]}
```

```json
{"scene":"office","classes":["Small office objects","Large static wall furniture","Large office objects","Bathroom","Bedroom","Chairs","Signs"],"object_semantic_ids":[2,3,4,8,9,11,15]}
```

Each JSON must include the corresponding official label-space hash and common-v2 alias-map hash. Each TXT file contains exactly the JSON `classes`, one line per class. The cache manifest binds only RGB-D/export inputs and never a target, GT, or prediction path.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest -q tests/datasets/test_tesse_cd.py \
  tests/evaluation/test_oviv2_tesse_cache_config.py
git add src/datasets/tesse_cd.py configs/evaluation tests/datasets \
  tests/evaluation/test_oviv2_tesse_cache_config.py
git commit -m "feat: add validated TESSE-CD RGB-D inputs"
```

Expected: all tests pass and no config contains a GT/target path.

### Task 5: Rename Active User-Facing OVIOVO Results To OVIV2

**Files:**
- Modify: `tools/benchmark_tables.py`
- Modify: `tools/import_benchmark_results.py`
- Modify: `docs/paper/benchmark_tokens.tsv`
- Modify: `docs/paper/benchmark_tables.md`
- Modify: `docs/paper/benchmark_tables.tex`
- Modify: `docs/paper/benchmark_tables_baselines.md`
- Modify: `docs/paper/benchmark_tables_baselines.tex`
- Modify: `tests/test_benchmark_table_package.py`
- Modify: `tests/evaluation/test_import_benchmark_results.py`

- [ ] **Step 1: Write failing registry and importer tests**

Require all 72 active user-facing method tokens to use OVIV2 naming: T2 has 10 `T2_OVIV2_*`, T4 has 24 renamed method tokens, S1 has 14, S2 has 12, and S3 has 12. Require zero active token beginning with `T2_OVIOVO_` and require T2 display mode `online`.

Add importer cases that accept a hash-bound `T2_OVIV2_CURRENT_MIOU`, accept six finite or source-bound unavailable official metrics, and reject every `T2_OVIOVO_*` binding.

Run:

```bash
pytest -q tests/test_benchmark_table_package.py \
  tests/evaluation/test_import_benchmark_results.py
```

Expected: tests fail on the legacy T2 registry and missing OVIV2 whitelist.

- [ ] **Step 2: Update generator names without touching legacy internals**

For T2 use:

```python
("OVIV2", "OVIV2", "online")
```

Update active T4/S1/S2/S3 keys and display text from OVIOVO to OVIV2, including `OVIV2_STATIC` where the existing key is `OVIOVO_STATIC`. Do not rename `src/evaluation/exporters/oviovo.py`, legacy pipeline imports/loggers, historical status reports, cache/environment paths, or legacy rejection fixtures.

- [ ] **Step 3: Add strict unavailable-binding import support**

Port only the reviewed helpers that require the evidence file to exist and match SHA-256:

```python
def _require_hashed_file(record: Mapping[str, object], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    _require_file(path)
    if record.get("sha256") != _sha256(path):
        raise ImportFailure(f"{label} hash mismatch")
    return path
```

Reject duplicate finite/unavailable bindings, empty reasons, unhashed evidence, method/split mismatch, and `OVIOVO`. Permit source-bound `N/A` only when the finalizer emitted a verified unavailable binding.

- [ ] **Step 4: Update only the active template rows and generated views**

Change the 10 T2 registry rows to `T2_OVIV2_*`/`method=OVIV2`. Apply the corresponding active T4/S1/S2/S3 rename. Update the four checked table views without regenerating verified numeric rows with `write --force`.

- [ ] **Step 5: Run the table/import gate and commit**

Run:

```bash
pytest -q tests/test_benchmark_table_package.py \
  tests/evaluation/test_import_benchmark_results.py
python tools/benchmark_tables.py check --output-dir docs/paper
git add tools docs/paper/benchmark_tokens.tsv docs/paper/benchmark_tables.md \
  docs/paper/benchmark_tables.tex docs/paper/benchmark_tables_baselines.md \
  docs/paper/benchmark_tables_baselines.tex tests/test_benchmark_table_package.py \
  tests/evaluation/test_import_benchmark_results.py
git commit -m "refactor: unify active benchmark naming as OVIV2"
```

Expected: tests and table check pass; verified baseline values remain unchanged.

### Task 6: Extract Shared Stage3 Config And Immutable Commit APIs

**Files:**
- Create: `src/oviv2/runner_config.py`
- Modify: `scripts/run_oviv2_replica.py`
- Modify: `src/oviv2/snapshot.py`
- Modify: `src/oviv2/runtime.py`
- Modify: `tests/evaluation/test_run_oviv2_replica_cli.py`
- Modify: `tests/oviv2/test_snapshot.py`

- [ ] **Step 1: Write failing config and non-overwrite tests**

Require:

```python
runtime = runtime_config_from_json({"ownership_min_net_support": 0.25})
assert runtime.ownership_min_net_support == 0.25

first = runtime.commit_new(tmp_path / "checkpoint")
before = tree_hash(tmp_path / "checkpoint")
with pytest.raises(FileExistsError):
    runtime.commit_new(tmp_path / "checkpoint")
assert tree_hash(tmp_path / "checkpoint") == before
```

Run:

```bash
pytest -q tests/oviv2/test_snapshot.py \
  tests/evaluation/test_run_oviv2_replica_cli.py
```

Expected: the parser ignores the ownership field and `commit_new` is absent.

- [ ] **Step 2: Extract the shared parsers**

Implement:

```python
def runtime_config_from_json(config: Mapping[str, Any]) -> Oviv2RuntimeConfig:
    """Parse the existing Replica Stage3 runtime fields plus ownership release."""

def semantic_fusion_config_from_json(
    config: Mapping[str, Any],
) -> SemanticFusionConfig | None:
    mode = config.get("fusion_semantic_mode", "disabled")
    if mode == "disabled":
        if "fusion_entity_weight_scale" in config:
            raise ValueError(
                "fusion_entity_weight_scale requires fusion_semantic_mode"
            )
        return None
    if mode != "uncertainty_linear":
        raise ValueError(
            "fusion_semantic_mode must be disabled or uncertainty_linear"
        )
    if config.get("dense_semantic_mode", "disabled") != "cached_probabilities":
        raise ValueError("uncertainty_linear fusion requires cached dense semantics")
    return SemanticFusionConfig(
        entity_weight_scale=config.get("fusion_entity_weight_scale", 0.5)
    )

def structure_config_from_json(
    config: Mapping[str, Any], *, voxel_size_m: float
) -> DepthStructureConfig:
    return DepthStructureConfig(
        enabled=bool(config.get("structure_enabled", True)),
        voxel_size_m=voxel_size_m,
        pixel_stride=int(
            config.get("structure_pixel_stride", config.get("pixel_stride", 4))
        ),
        min_valid_points=int(
            config.get("structure_min_valid_points", config.get("min_valid_points", 10))
        ),
        horizontal_threshold=float(config.get("structure_horizontal_threshold", 0.6)),
        wall_vertical_threshold=float(config.get("structure_wall_vertical_threshold", 0.5)),
        min_component_pixels=int(config.get("structure_min_component_pixels", 500)),
        min_component_fraction=float(config.get("structure_min_component_fraction", 0.01)),
        max_components_per_class=int(config.get("structure_max_components_per_class", 5)),
        object_exclusion_dilation=int(config.get("structure_object_exclusion_dilation", 3)),
        wall_confidence=float(config.get("structure_wall_confidence", 0.75)),
        floor_confidence=float(config.get("structure_floor_confidence", 0.85)),
        ceiling_confidence=float(config.get("structure_ceiling_confidence", 0.80)),
    )
```

Move the complete current `_runtime_config` body from `scripts/run_oviv2_replica.py` into `runtime_config_from_json`, retaining every field and default, and add this argument to its `Oviv2RuntimeConfig` constructor:

```python
ownership_min_net_support=float(config.get("ownership_min_net_support", 1e-6)),
```

Keep the Replica runner's private function names as thin wrappers calling these functions so existing CLI behavior stays byte-compatible.

- [ ] **Step 3: Add atomic non-overwriting snapshot publication**

Implement the `VoxelMapSnapshot.commit_new` and `Oviv2Runtime.commit_new` methods. Write the snapshot into a sibling temporary directory, fsync files and directory, then atomically publish only if the target does not exist. If Linux `renameat2(RENAME_NOREPLACE)` is unavailable, create the target with an exclusive lock directory before rename. Always raise `FileExistsError` without changing the existing target.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
pytest -q tests/oviv2/test_snapshot.py tests/oviv2/test_runtime.py \
  tests/evaluation/test_run_oviv2_replica_cli.py
git add src/oviv2/runner_config.py src/oviv2/snapshot.py src/oviv2/runtime.py \
  scripts/run_oviv2_replica.py tests/oviv2/test_snapshot.py \
  tests/evaluation/test_run_oviv2_replica_cli.py
git commit -m "feat: add reusable Stage3 runtime configuration"
```

Expected: all selected tests pass, including unchanged Replica runner fixtures.

### Task 7: Build The TESSE Object Frontend Orchestrator

**Files:**
- Create: `scripts/precompute_oviv2_tesse_frontend.py`
- Create: `configs/oviv2_tesse_cd_frontend_stage3.json`
- Test: `tests/evaluation/test_oviv2_tesse_frontend_orchestrator.py`

- [ ] **Step 1: Write failing command and manifest tests**

Require separate Apartment/Office commands, GPUs 0/1, frame counts 1745/4346, 720x480 source size, independent vocabulary hashes, model hashes, and unknown-scene rejection. Validate every cache record has:

```python
required = {"mask", "xyxy", "confidence", "class_id", "classes", "image_feats", "text_feats"}
assert required <= set(cache_payload)
assert cache_payload["mask"].shape[-2:] == (480, 720)
```

Run: `pytest -q tests/evaluation/test_oviv2_tesse_frontend_orchestrator.py`

Expected: import fails because the orchestrator does not exist.

- [ ] **Step 2: Implement deterministic commands and validation**

Add `FrontendCommand`, `build_commands()`, `build_environment()`, `warm_shared_clip_cache()`, `validate_frontend_cache()`, and `main()`. Invoke the existing `streamlined_detections.py` with the frozen scene TXT vocabulary, YOLO model `/home/ww/vv/paper2/DualMap/model/yolov8l-world.pt`, MobileSAM `/home/ww/vv/paper2/DualMap/model/mobile_sam.pt`, and frozen CLIP paths. Bind upstream script and all weight files by SHA-256 because the upstream directory is not a verifiable checkout.

Publish `frontend_manifest.json` only after all frame files validate, sorted source IDs equal `range(frame_count)`, and their prefix digest is recorded.

- [ ] **Step 3: Run frontend tests and commit**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_oviv2_tesse_frontend_orchestrator.py \
  tests/evaluation/test_oviv2_replica_frontend_orchestrator.py
git add scripts/precompute_oviv2_tesse_frontend.py \
  configs/oviv2_tesse_cd_frontend_stage3.json \
  tests/evaluation/test_oviv2_tesse_frontend_orchestrator.py
git commit -m "feat: precompute TESSE object observations"
```

Expected: tests pass without running full inference.

### Task 8: Extend RADSeg Dense Precompute To TESSE-CD

**Files:**
- Modify: `scripts/precompute_oviv2_dense_semantics.py`
- Create: `configs/oviv2_tesse_apartment_dense_stage3.json`
- Create: `configs/oviv2_tesse_office_dense_stage3.json`
- Modify: `tests/evaluation/test_precompute_oviv2_dense_semantics.py`

- [ ] **Step 1: Write failing TESSE and streaming tests**

Require `_preflight()` to accept `dataset == "TESSE-CD"`, use per-scene vocabulary, preserve contiguous source IDs, validate export manifest/camera values, and load only the requested RGB frame:

```python
preflight = _preflight(config, num_frames=5)
assert not hasattr(preflight, "rgb_frames")
rgb = _load_rgb_frame(preflight.dataset, 4)
assert rgb.shape == (480, 720, 3)
```

Run: `pytest -q tests/evaluation/test_precompute_oviv2_dense_semantics.py`

Expected: TESSE-CD is rejected as an unsupported dataset.

- [ ] **Step 2: Add the TESSE dataset branch**

Use `TesseCdRgbdDataset` instead of materializing all RGB frames. Read vocabulary and object IDs from the scene record, keep the existing worker protocol unchanged, and retain the frozen RADSeg output contract: `class_ids/probabilities` shape `(120, 180, 4)` and `entropy/margin` shape `(120, 180)`.

- [ ] **Step 3: Run dense tests and commit**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-radseg/bin/python -m pytest -q \
  tests/evaluation/test_precompute_oviv2_dense_semantics.py \
  tests/oviv2/test_dense_semantics.py tests/oviv2/test_radseg_dense_worker.py
git add scripts/precompute_oviv2_dense_semantics.py \
  configs/oviv2_tesse_apartment_dense_stage3.json \
  configs/oviv2_tesse_office_dense_stage3.json \
  tests/evaluation/test_precompute_oviv2_dense_semantics.py
git commit -m "feat: precompute TESSE dense semantics"
```

Expected: all tests pass and the worker code remains unchanged.

### Task 9: Export Stage3 Snapshots As Neutral Current Maps

**Files:**
- Create: `src/evaluation/oviv2_tesse.py`
- Modify: `scripts/evaluation/export_tesse_temporal_artifact.py`
- Create: `tests/evaluation/test_oviv2_tesse.py`
- Modify: `tests/evaluation/test_export_tesse_temporal_artifact.py`

- [ ] **Step 1: Write failing ownership/export tests**

Test a snapshot with owned, released, unknown, and structural vertices. Require owned object vertices to become deterministic entities; released, structure, unknown, and unowned vertices must enter `background_xyz`. Require no historical registry entity without current owned vertices.

```python
neutral = build_neutral_current_snapshot(
    snapshot,
    timestamp_ns=123,
    class_names=("unknown", "chair"),
    object_semantic_ids=frozenset({1}),
    fusion=SemanticFusionConfig(entity_weight_scale=0.49),
    timestamp_ns_by_frame=(100, 123),
)
assert [entity.entity_id for entity in neutral.entities] == ["oviv2:7:semantic:1"]
assert neutral.background_xyz.shape[1] == 3
```

Run: `pytest -q tests/evaluation/test_oviv2_tesse.py`

Expected: import fails because the exporter adapter is absent.

- [ ] **Step 2: Implement schedule and neutral snapshot contracts**

Add:

```python
@dataclass(frozen=True)
class TesseCausalCheckpoint:
    frame_index: int
    timestamp_ns: int
    relative_timestamp_ns: int
    event_ids: Sequence[str]
    roles: Sequence[str]

def load_causal_checkpoints(
    schedule: Path, *, scene: str, frame_count: int
) -> Sequence[TesseCausalCheckpoint]:
    """Load, sort, and validate every scheduled checkpoint for one scene."""

def build_neutral_current_snapshot(
    snapshot: VoxelMapSnapshot,
    *,
    timestamp_ns: int,
    class_names: Sequence[str],
    object_semantic_ids: frozenset[int],
    fusion: SemanticFusionConfig,
    timestamp_ns_by_frame: Sequence[int],
) -> MapSnapshot:
    """Convert immutable Stage3 state into the neutral current-map contract."""
```

Replace the docstring-only plan signatures with validation and deterministic conversion code. `load_causal_checkpoints` rejects duplicate/out-of-range frames, non-increasing timestamps, unknown scenes, and count mismatches. `build_neutral_current_snapshot` uses `derive_labeled_mesh`, snapshot registry posteriors, current ownership only, and stable `(entity_id, semantic_id, voxel_key)` ordering.

- [ ] **Step 3: Make temporal artifacts repeat-root independent**

Copy source sidecars into each artifact and store paths relative to the temporal root. Canonical JSON must use sorted keys, finite floats, separators `(",", ":")`, and one trailing newline. Exclude absolute output roots, command lines, host/GPU data, and wall-clock timings from byte-compared artifacts.

- [ ] **Step 4: Run exporter tests and commit**

Run:

```bash
pytest -q tests/evaluation/test_oviv2_tesse.py \
  tests/evaluation/test_export_tesse_temporal_artifact.py
git add src/evaluation/oviv2_tesse.py \
  scripts/evaluation/export_tesse_temporal_artifact.py \
  tests/evaluation/test_oviv2_tesse.py \
  tests/evaluation/test_export_tesse_temporal_artifact.py
git commit -m "feat: export OVIV2 causal current maps"
```

Expected: two artifacts created under different roots are byte-identical.

### Task 10: Implement The Full-Frame TESSE Online Runner

**Files:**
- Create: `scripts/evaluation/run_oviv2_tesse_cd.py`
- Create: `configs/oviv2_tesse_cd_apartment_v1.json`
- Create: `configs/oviv2_tesse_cd_office_v1.json`
- Test: `tests/evaluation/test_run_oviv2_tesse_cd.py`

- [ ] **Step 1: Write the failing five-frame causal test**

Use dependency injection for runtime and cache loaders. Require the observed event order:

```python
assert calls == [
    "load:0", "process:0",
    "load:1", "process:1", "commit:1", "export:1",
    "load:2", "process:2",
    "load:3", "process:3", "commit:3", "export:3",
    "load:4", "process:4",
]
```

Require `processed_frame_count == 5` even though the final checkpoint is frame 3. Require `consumed_through_frame=t`, `consumed_through_frame_exclusive=t+1`, true ns timestamps, missing/duplicate/extra checkpoint rejection, and byte-identical neutral outputs across two roots.

Run: `pytest -q tests/evaluation/test_run_oviv2_tesse_cd.py`

Expected: import fails because the runner is absent.

- [ ] **Step 2: Build only the Stage3 runtime path**

Parse only `--config` and `--output` in production mode; the scene is an immutable field in the selected scene config. Construct `Oviv2Runtime` through `runtime_config_from_json`, `DepthStructureConfig`, cached YOLO-World/MobileSAM observations, RADSeg dense caches, signed visibility, reversible ownership, and semantic fusion. Do not import route3 surface composition or any ScanNet200 Stage4 module.

- [ ] **Step 3: Commit immutable scheduled checkpoints**

For each source frame execute exactly:

```python
frame = dataset[frame_index]
observations = frontend.load(frame_index)
dense = dense_cache.load(frame_index)
runtime.process_frame(frame, observations=observations, dense_semantics=dense)
if frame_index in checkpoints:
    snapshot = runtime.commit_new(checkpoint_voxel_dir)
    export_checkpoint(snapshot, checkpoint, neutral_source_dir)
```

Write checkpoint status with both causal boundaries, timestamp, event IDs, and roles. Reject output reuse and publish `run_manifest.json` only after all 1745 or 4346 frames and every scheduled checkpoint complete.

- [ ] **Step 4: Separate deterministic output from machine provenance**

`run_manifest.json` binds normalized config, source/export/camera/trajectory/timestamps, frontend/dense caches, schedule, checkpoint artifacts, and maintenance parameters. Write absolute paths, command, Python/CUDA/library/GPU/host facts to `run_provenance.json`; write wall time only to `timing.json`. Neither file participates in the repeat byte comparison.

- [ ] **Step 5: Run runner tests and commit**

Run:

```bash
pytest -q tests/evaluation/test_run_oviv2_tesse_cd.py \
  tests/evaluation/test_oviv2_tesse.py tests/oviv2/test_runtime.py
git add scripts/evaluation/run_oviv2_tesse_cd.py \
  configs/oviv2_tesse_cd_apartment_v1.json \
  configs/oviv2_tesse_cd_office_v1.json \
  tests/evaluation/test_run_oviv2_tesse_cd.py
git commit -m "feat: run OVIV2 online on TESSE-CD"
```

Expected: all tests pass and an AST/import assertion confirms no route3 or Stage4 dependency.

### Task 11: Build Prediction-Independent Occlusion Evidence

**Files:**
- Create: `scripts/evaluation/derive_tesse_cd_occlusion_v1.py`
- Create: `src/evaluation/oviv2_occlusion.py`
- Create: `configs/evaluation/manifests/tesse_cd_occlusion_v1.json`
- Create: `scripts/evaluation/evaluate_oviv2_tesse_occlusion.py`
- Modify: `src/oviv2/runtime.py`
- Modify: `scripts/evaluation/run_oviv2_tesse_cd.py`
- Modify: `tests/oviv2/test_visibility.py`
- Modify: `tests/oviv2/test_runtime.py`
- Create: `tests/evaluation/test_derive_tesse_cd_occlusion_v1.py`
- Create: `tests/evaluation/test_oviv2_occlusion.py`
- Create: `tests/evaluation/test_evaluate_oviv2_tesse_occlusion.py`

- [ ] **Step 1: Write failing target classification tests**

For a projected GT surface point require:

```python
assert classify_depth(d_obs=2.00, d_gt=2.05, tolerance_m=0.10) == "present"
assert classify_depth(d_obs=1.50, d_gt=2.05, tolerance_m=0.10) == "occluded"
assert classify_depth(d_obs=2.50, d_gt=2.05, tolerance_m=0.10) == "absent"
```

Reject RGB, method artifacts, predictions, future lifecycle intervals, changed source hashes, and a scene with no qualifying episode.

- [ ] **Step 2: Derive hash-bound occlusion episodes from GT/depth only**

For every official DSG object active at a timestamp, intersect a prior present anchor voxel set with current occluded voxels. Merge consecutive non-empty frames into episodes. Store scene, object ID, lifecycle interval, anchor, start/end, checkpoint list, and occlusion fraction in JSON; store voxel keys as sorted `N x 3 int64` arrays. Set `prediction_inputs_used=false` and use an exact source-file allowlist.

- [ ] **Step 3: Pre-register the stress layer**

Report all episodes and thresholds 0.50, 0.75, and 0.90; define 0.90 as the headline stress layer. Fail closed if either scene has zero 0.90 episodes. Run the derivation twice in temporary roots and require byte-identical manifest and NPZ hashes.

- [ ] **Step 4: Run target tests**

Run:

```bash
pytest -q tests/evaluation/test_derive_tesse_cd_occlusion_v1.py
```

Expected: all target, leak, hash, and determinism cases pass.

- [ ] **Step 5: Commit the target package code**

```bash
git add scripts/evaluation/derive_tesse_cd_occlusion_v1.py \
  src/evaluation/oviv2_occlusion.py \
  configs/evaluation/manifests/tesse_cd_occlusion_v1.json \
  tests/evaluation/test_derive_tesse_cd_occlusion_v1.py
git commit -m "feat: derive TESSE occlusion evidence targets"
```

- [ ] **Step 6: Write failing runtime ablation tests**

Run the same owned entity through 1, 10, and 100 occluded frames. Under `signed_depth`, negative support, owner, epoch, and revision remain unchanged. Under `missing_as_absence`, projectable present/occluded voxels receive negative support and eventually release ownership. `UNOBSERVED` voxels never receive this ablation penalty.

- [ ] **Step 7: Add the frozen visibility policy**

Add to `Oviv2RuntimeConfig`:

```python
missing_observation_policy: Literal["signed_depth", "missing_as_absence"] = "signed_depth"
```

Keep current behavior for `signed_depth`: only `ABSENT` adds negative entity support. In the ablation, apply identical negative support to projectable `PRESENT | OCCLUDED` voxels for an entity with no matched observation; never penalize `UNOBSERVED`.

- [ ] **Step 8: Implement fixed-anchor ownership metrics**

At the anchor snapshot, map each GT object to the majority predicted owner on anchor-owned target voxels and never rematch later. Report finite counts and rates:

```python
false_release_rate = false_release_count / anchor_owned_target_voxels
false_reassignment_rate = false_reassignment_count / anchor_owned_target_voxels
retained_ownership_recall = retained_owner_count / anchor_owned_target_voxels
gt_retained_object_recall = retained_owner_count / all_gt_occluded_target_voxels
```

Also report voxel/object/episode counts and zero-release episode rate. Reject future snapshots, missing checkpoints, owner ties without the deterministic tie rule, and source hash mismatch.

- [ ] **Step 9: Run runtime and evaluator tests**

Run:

```bash
pytest -q tests/oviv2/test_visibility.py tests/oviv2/test_runtime.py \
  tests/oviv2/test_ownership.py tests/evaluation/test_oviv2_occlusion.py \
  tests/evaluation/test_evaluate_oviv2_tesse_occlusion.py
```

Expected: all tests pass; signed-depth and ablation differ on the synthetic occlusion sequence.

- [ ] **Step 10: Commit the runtime evidence path**

```bash
git add src/oviv2/runtime.py scripts/evaluation/run_oviv2_tesse_cd.py \
  scripts/evaluation/evaluate_oviv2_tesse_occlusion.py \
  tests/oviv2/test_visibility.py tests/oviv2/test_runtime.py \
  tests/evaluation/test_oviv2_occlusion.py \
  tests/evaluation/test_evaluate_oviv2_tesse_occlusion.py
git commit -m "feat: evaluate OVIV2 occlusion retention"
```

### Task 12: Bridge Neutral Checkpoints Into The Official TESSE Evaluator

**Files:**
- Create: `src/evaluation/baselines/tesse_cd.py`
- Create: `scripts/evaluation/prepare_temporal_khronos_bridge.py`
- Create: `scripts/evaluation/compat/khronos_temporal_bridge/import_temporal_baseline.cpp`
- Create: `scripts/evaluation/run_temporal_khronos_bridge.py`
- Create: `scripts/evaluation/run_khronos_official_eval.py`
- Create: `tests/evaluation/test_tesse_cd_metrics.py`
- Create: `tests/evaluation/test_prepare_temporal_khronos_bridge.py`
- Create: `tests/evaluation/test_run_temporal_khronos_bridge.py`
- Create: `tests/evaluation/test_run_khronos_official_eval.py`

- [ ] **Step 1: Write failing neutral bridge tests**

Build a three-checkpoint `OVIV2` temporal fixture with stable, disappearing, and reappearing entity IDs. Require stable Khronos node symbols ordered by first appearance then entity ID, closed-open presence intervals, official semantic IDs, causal trajectory prefixes, and no synthetic trajectory for an entity with fewer than two native samples.

Run:

```bash
pytest -q tests/evaluation/test_tesse_cd_metrics.py \
  tests/evaluation/test_prepare_temporal_khronos_bridge.py \
  tests/evaluation/test_run_temporal_khronos_bridge.py \
  tests/evaluation/test_run_khronos_official_eval.py
```

Expected: imports fail because the bridge and metric summarizer have not been ported.

- [ ] **Step 2: Port the reviewed bridge as an OVIV2-neutral adapter**

Port only the four reviewed production files and their focused tests from `tsdf-contamination-visibility-gates`. Accept temporal identity:

```python
if temporal.get("dataset") != "TESSE-CD":
    raise ValueError("official bridge requires TESSE-CD")
if temporal.get("method") != "OVIV2":
    raise ValueError("official bridge requires the frozen OVIV2 artifact")
if temporal.get("mode") != "causal_checkpoints":
    raise ValueError("official bridge requires causal checkpoints")
```

Map only neutral snapshot entities and background into the importer. Use official scene label-space YAML files, stable `O{index}` symbols, checkpoint-bounded presence intervals, and native per-frame centroids emitted by the OVIV2 runner. Do not read live runtime state, common-v2 targets, change annotations, or future trajectory samples.

- [ ] **Step 3: Make the Khronos importer build hash-bound and isolated**

Stage `import_temporal_baseline.cpp` into `/home/ww/oviovo_baseline_builds/khronos-jazzy-ws/src/khronos/khronos_eval/app/`, add the bounded CMake block once, build only `khronos_eval`, and record source/CMake/executable hashes. The adapter run mode is `causal_checkpoints`, while the paper display mode remains `online`.

- [ ] **Step 4: Wrap the official evaluator and summarize repeats**

Patch only the evaluator's GT paths from the canonical TESSE manifest. Invoke `evaluate_pipeline.sh` on the imported `.4dmap`, require clean completion, and hash `static_objects.csv`, `dynamic_objects.csv`, and `background_mesh.csv`. Produce scene metric JSON with:

```json
{"dataset":"TESSE-CD","method":"OVIV2","mode":"causal_checkpoints","metrics":{"object_f1":0.0,"dynamic_f1":0.0,"change_f1":0.0}}
```

The numeric zeros are a fixture shape, not experiment results. In production, parse the official CSV aggregation. If upstream omits a defined metric, emit a hashed partial/unavailable record with the exact upstream reason. Write the metric summary twice and require byte identity.

- [ ] **Step 5: Run official bridge tests and commit**

Run:

```bash
pytest -q tests/evaluation/test_tesse_cd_metrics.py \
  tests/evaluation/test_prepare_temporal_khronos_bridge.py \
  tests/evaluation/test_run_temporal_khronos_bridge.py \
  tests/evaluation/test_run_khronos_official_eval.py
git add src/evaluation/baselines/tesse_cd.py \
  scripts/evaluation/prepare_temporal_khronos_bridge.py \
  scripts/evaluation/compat/khronos_temporal_bridge/import_temporal_baseline.cpp \
  scripts/evaluation/run_temporal_khronos_bridge.py \
  scripts/evaluation/run_khronos_official_eval.py tests/evaluation
git commit -m "feat: evaluate OVIV2 with official TESSE metrics"
```

Expected: focused tests pass and no bridge code imports OVIV2 runtime internals.

### Task 13: Merge Parallel Lanes And Run The Integration Gate

**Files:**
- Verify all files from Tasks 2-11.

- [ ] **Step 1: Integrate lane commits in dependency order**

Cherry-pick wave A in this order: protocol/finalizers, dataset, naming, runtime foundation. Run their focused gates. Create wave B lane worktrees from that integration commit, then cherry-pick frontend, dense, neutral export, and occlusion-target commits. Run their focused gates. Create wave C lane worktrees from the wave-B integration commit, then cherry-pick online runner, occlusion-runtime, and official-bridge commits. Resolve no conflict by deleting another lane's assertions. If a conflict appears in an exclusive file, stop and have that file's owner reconcile it.

- [ ] **Step 2: Run formatting/static repository checks**

Run:

```bash
git diff --check 47962fb..HEAD
python tools/benchmark_tables.py check --output-dir docs/paper
python -m compileall -q src scripts tools
```

Expected: all commands exit 0.

- [ ] **Step 3: Run the complete focused suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q \
  tests/datasets/test_tesse_cd.py \
  tests/oviv2/test_runtime.py tests/oviv2/test_snapshot.py \
  tests/oviv2/test_visibility.py tests/oviv2/test_ownership.py \
  tests/evaluation/test_evaluation_package_imports.py \
  tests/evaluation/test_tesse_cd_manifest.py \
  tests/evaluation/test_derive_tesse_cd_causal_schedule.py \
  tests/evaluation/test_derive_tesse_cd_common_v2.py \
  tests/evaluation/test_evaluate_tesse_cd_common_v2.py \
  tests/evaluation/test_finalize_tesse_common_v2.py \
  tests/evaluation/test_finalize_tesse_t2.py \
  tests/evaluation/test_oviv2_tesse_cache_config.py \
  tests/evaluation/test_oviv2_tesse_frontend_orchestrator.py \
  tests/evaluation/test_precompute_oviv2_dense_semantics.py \
  tests/evaluation/test_oviv2_tesse.py \
  tests/evaluation/test_run_oviv2_tesse_cd.py \
  tests/evaluation/test_prepare_temporal_khronos_bridge.py \
  tests/evaluation/test_run_temporal_khronos_bridge.py \
  tests/evaluation/test_run_khronos_official_eval.py \
  tests/evaluation/test_derive_tesse_cd_occlusion_v1.py \
  tests/evaluation/test_evaluate_oviv2_tesse_occlusion.py \
  tests/evaluation/test_import_benchmark_results.py \
  tests/test_benchmark_table_package.py
```

Expected: all tests pass. Investigate failures with `superpowers:systematic-debugging` before changing implementation.

- [ ] **Step 4: Run a five-frame end-to-end smoke test twice**

Use a dedicated fixture config that is unavailable from production CLI and compare the two temporal trees:

```bash
pytest -q tests/evaluation/test_run_oviv2_tesse_cd.py::test_five_frame_end_to_end_is_byte_identical
```

Expected: PASS and identical relative file lists and SHA-256 values.

### Task 14: Generate And Validate Frozen Semantic Caches

**Files:**
- Generated: `/home/ww/oviovo_frontend_cache/tesse_cd_native_v1/{apartment,office}/`
- Generated: `/home/ww/oviovo_dense_cache/tesse_cd_radseg_b_sam_s4_k4_native/{apartment,office}/`

- [ ] **Step 1: Preflight disk, GPUs, weights, and canonical inputs**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
df -h /home/ww
python scripts/precompute_oviv2_tesse_frontend.py \
  --config configs/oviv2_tesse_cd_frontend_stage3.json --preflight-only
```

Expected: three A40 GPUs are visible, at least 300 GiB remain, and every source/model hash matches.

- [ ] **Step 2: Generate object caches in parallel on GPUs 0 and 1**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/precompute_oviv2_tesse_frontend.py \
  --config configs/oviv2_tesse_cd_frontend_stage3.json \
  --gpu 0 --gpu 1
```

Expected: Apartment publishes 1745 frame caches and Office publishes 4346; both manifests validate 480x720 masks and frozen weight/vocabulary hashes.

- [ ] **Step 3: Generate both dense caches in parallel on GPUs 1 and 2**

Run from Bash:

```bash
RADSEG_COMMON=(
  --backend radseg
  --source-root /home/ww/oviovo_references/modules/RADSeg
  --radio-root /home/ww/oviovo_references/modules/RADIO
  --model-version /home/ww/oviovo_benchmark_assets/weights/radio/c-radio_v3-b_half-44653a.pth.tar
  --lang-model siglip2
  --language-model-root /home/ww/oviovo_benchmark_assets/weights/siglip2-so400m-patch16-naflex-cc24074
  --language-model-id google/siglip2-so400m-patch16-naflex
  --language-model-revision cc24074f717b612951c2dead130904ab9b65a81e
  --language-model-sha256 0e5dbd4cd9511c4335ae4a144ac5187841a2f7fed2a07df9c36668516e02cafe
  --device cuda --sample-stride 4 --top-k 4 --amp --sam-refinement
  --sam-checkpoint /home/ww/oviovo_benchmark_assets/weights/segment_anything/sam_vit_h_4b8939.pth
)
CUDA_VISIBLE_DEVICES=1 /home/ww/miniconda3/envs/oviovo-radseg/bin/python \
  scripts/precompute_oviv2_dense_semantics.py \
  --config configs/oviv2_tesse_apartment_dense_stage3.json \
  --output /home/ww/oviovo_dense_cache/tesse_cd_radseg_b_sam_s4_k4_native/apartment \
  --num-frames 1745 \
  --classes-json configs/evaluation/vocabularies/tesse_cd_apartment.json \
  "${RADSEG_COMMON[@]}" &
APARTMENT_PID=$!
CUDA_VISIBLE_DEVICES=2 /home/ww/miniconda3/envs/oviovo-radseg/bin/python \
  scripts/precompute_oviv2_dense_semantics.py \
  --config configs/oviv2_tesse_office_dense_stage3.json \
  --output /home/ww/oviovo_dense_cache/tesse_cd_radseg_b_sam_s4_k4_native/office \
  --num-frames 4346 \
  --classes-json configs/evaluation/vocabularies/tesse_cd_office.json \
  "${RADSEG_COMMON[@]}" &
OFFICE_PID=$!
wait "$APARTMENT_PID"
wait "$OFFICE_PID"
```

Expected: both commands exit 0; manifest class counts are 10/7 and all per-frame checksums validate. Never run object and dense workers concurrently on the same GPU.

- [ ] **Step 4: Hash and lock cache manifests**

Run the preflight/validation mode for all four caches and save their manifest SHA-256 values for the freeze manifest. A completed cache has no partial marker and contains exactly the declared source IDs.

### Task 15: Tune Apartment Maintenance Parameters And Freeze v1

**Files:**
- Create: `scripts/evaluation/freeze_oviv2_tesse_cd.py`
- Create: `tests/evaluation/test_freeze_oviv2_tesse_cd.py`
- Generated: `configs/oviv2_tesse_cd_apartment_v1_frozen.json`
- Generated: `configs/oviv2_tesse_cd_office_v1_frozen.json`
- Generated: `outputs/oviv2-tessecd-v1/freeze_manifest.json`

- [ ] **Step 1: Write failing freeze invariants**

Require a clean commit, lineage to `47962fb`, identical normalized algorithm hashes across scenes, exact input/cache/model/protocol hashes, and differences limited to scene/path/vocabulary/source manifest fields. Reject a dirty tree, missing hash, different Office algorithm value, route3 config/import, or Stage4 config/import.

- [ ] **Step 2: Implement the freeze command**

The command records clean commit/parent lineage, normalized algorithm and per-scene config hashes, frontend/dense/model hashes, RGB-D/source DB/trajectory/timestamp/camera hashes, schedule/target/label/alias/evaluator/finalizer hashes, environment facts, and exact run commands. It must refuse an output directory containing any prior run.

- [ ] **Step 3: Run the predeclared Apartment sweep**

Only vary:

```text
visibility_depth_tolerance_m in [0.05, 0.10, 0.15]
absence_negative_support in [0.5, 1.0]
ownership_min_net_support in [0.000001, 0.5, 1.0]
```

Keep all detection, association, tracking, TSDF, dense semantic, fusion, and meshing values identical to `configs/oviv2_replica8_stage3.json`. Evaluate all 18 candidates only on Apartment common-v2 targets, in batches of at most three concurrent mapping processes after cache generation. Select by the predeclared lexicographic rule: maximize current mIoU, minimize ghost rate, maximize background F@5 cm, minimize recovery frames; break an exact tie by smaller config hash. Save every candidate summary and hash.

- [ ] **Step 4: Freeze before any Office run**

Run:

```bash
python scripts/evaluation/freeze_oviv2_tesse_cd.py \
  --apartment-selection outputs/oviv2-tessecd-tuning/apartment/selection.json \
  --apartment-config configs/oviv2_tesse_cd_apartment_v1.json \
  --office-config configs/oviv2_tesse_cd_office_v1.json \
  --output-apartment-config configs/oviv2_tesse_cd_apartment_v1_frozen.json \
  --output-office-config configs/oviv2_tesse_cd_office_v1_frozen.json \
  --output-manifest outputs/oviv2-tessecd-v1/freeze_manifest.json
```

Expected: PASS, a clean freeze commit is recorded, and no Office metric source exists before the freeze timestamp/commit.

- [ ] **Step 5: Commit frozen code/config and rerun freeze**

```bash
git add scripts/evaluation/freeze_oviv2_tesse_cd.py \
  tests/evaluation/test_freeze_oviv2_tesse_cd.py \
  configs/oviv2_tesse_cd_apartment_v1_frozen.json \
  configs/oviv2_tesse_cd_office_v1_frozen.json
git commit -m "chore: freeze oviv2-tessecd-v1"
git status --short
```

Expected: empty status. Regenerate the freeze manifest so its commit points to this clean commit.

### Task 16: Run Both Scenes Twice And Evaluate Determinism

**Files:**
- Generated: `outputs/oviv2-tessecd-v1/{apartment,office}/run{1,2}/`
- Generated: `outputs/oviv2-tessecd-v1/evaluation/{apartment,office}/run{1,2}.json`

- [ ] **Step 1: Run Apartment and Office first repeats in parallel only after freeze**

Run:

```bash
python scripts/evaluation/run_oviv2_tesse_cd.py \
  --config configs/oviv2_tesse_cd_apartment_v1_frozen.json \
  --output outputs/oviv2-tessecd-v1/apartment/run1 &
APARTMENT_PID=$!
python scripts/evaluation/run_oviv2_tesse_cd.py \
  --config configs/oviv2_tesse_cd_office_v1_frozen.json \
  --output outputs/oviv2-tessecd-v1/office/run1 &
OFFICE_PID=$!
wait "$APARTMENT_PID"
wait "$OFFICE_PID"
```

Expected: both commands exit 0, process exactly 1745/4346 frames, and publish exactly the scheduled immutable checkpoints. If RAM pressure exceeds the freeze preflight limit, run scenes serially without changing config.

- [ ] **Step 2: Run the second repeats from empty directories**

Repeat the commands with `run2`. Never copy, resume, or hard-link run1 runtime state into run2.

- [ ] **Step 3: Evaluate four temporal artifacts**

Run one evaluator process per scene/repeat; each invocation binds the common-v2 target package and writes canonical result JSON. Expected: all four summaries have `method=OVIV2`, `mode=causal_checkpoints`, finite metrics, and passing causal/source hash audits.

- [ ] **Step 4: Compare deterministic products**

Compare relative file lists and SHA-256 values for the byte-compared temporal artifacts and evaluator summaries:

```bash
cmp outputs/oviv2-tessecd-v1/evaluation/apartment/run1.json \
    outputs/oviv2-tessecd-v1/evaluation/apartment/run2.json
cmp outputs/oviv2-tessecd-v1/evaluation/office/run1.json \
    outputs/oviv2-tessecd-v1/evaluation/office/run2.json
```

Expected: both commands exit 0. Compare checkpoint content hashes recorded in temporal indices; machine provenance and timings are excluded.

- [ ] **Step 5: Run signed-depth and ablation occlusion evaluations**

Derive the occlusion-v1 target package once from GT/depth. Run the frozen signed-depth checkpoint stream and a separate `missing_as_absence` run with every other input/config value identical. Evaluate both with fixed-anchor mapping.

Expected: results are marked discriminative only if the ablation has a higher false-release rate or lower retained recall. The headline safety clause additionally requires both scenes to have 0.90 stress episodes and signed-depth `false_release_count=0`, `false_reassignment_count=0`, and anchor-normalized recall `=1.0`.

- [ ] **Step 6: Run the official evaluator bridge for both repeats**

Prepare the temporal Khronos bridge for each scene/repeat, import it into a separate `.4dmap`, and run the official evaluator. Require matching source run identity, exact official schedule coverage, no future trajectory samples, and byte-identical metric summaries between run1/run2.

Expected: Apartment and Office each produce finite official object/dynamic/change F1 or a precise hashed upstream-unavailable record accepted by the strict finalizer.

### Task 17: Strictly Finalize, Import Results, And Fill The Abstract

**Files:**
- Modify: `docs/paper/benchmark_tokens.tsv` through importer only for result values.
- Modify: active benchmark table views through the checked generator path.
- Modify in the original paper workspace after the frozen run: `/home/ww/.config/superpowers/worktrees/oviovo/oviv2-stage3-final-batch/docs/paper/oviv2_aaai/main.tex`
- Generated: `outputs/oviv2-tessecd-v1/final/common_v2.json`
- Generated: `outputs/oviv2-tessecd-v1/final/official_t2.json`
- Generated: `outputs/oviv2-tessecd-v1/final/occlusion_v1.json`

- [ ] **Step 1: Finalize common-v2 and official metrics**

Run the strict common finalizer with Apartment/Office run1 and repeat summaries. Run the official T2 finalizer with the official metric sources from the same frozen run identity. Finite official metrics become bindings; unavailable official metrics become hashed, source-bound unavailable bindings rather than invented values.

Expected: common finalizer emits four `T2_OVIV2_*` bindings, official finalizer accounts for six, no token appears twice, and both finalizers report PASS.

- [ ] **Step 2: Import all ten OVIV2 T2 bindings**

Run:

```bash
python tools/import_benchmark_results.py \
  --registry docs/paper/benchmark_tokens.tsv \
  --result outputs/oviv2-tessecd-v1/final/common_v2.json \
  --result outputs/oviv2-tessecd-v1/final/official_t2.json
python tools/benchmark_tables.py check --output-dir docs/paper
```

Expected: the 10 `T2_OVIV2_*` rows are VERIFIED or source-bound N/A, no `T2_OVIOVO_*` row exists, and all table views match the registry.

- [ ] **Step 3: Compute only protocol-eligible baseline deltas**

For each claimed metric, compare OVIV2 against the best eligible non-oracle row with a finalized common-v2 value. Exclude Khronos GT semantics and any unavailable/partial row. Record the baseline name, absolute delta, relative delta when the denominator is nonzero, direction, and source token IDs in the final result JSON.

- [ ] **Step 4: Fill the abstract with finalized evidence**

Replace the numerical placeholder with the finalized primary metrics and eligible-baseline deltas. Use the stale-state/recovery wording supported by current mIoU, ghost rate, background F@5 cm, and recovery frames. Include an occlusion clause only if Task 16's headline gate passes; phrase it as no observed false release on the reported GT-validated episodes, not a universal guarantee. Otherwise remove the occlusion clause.

The paper directory is existing untracked user work and is intentionally absent from the clean frozen-run branch. Apply only the evidence-bounded abstract edit to that original workspace after finalization; do not copy the paper tree into the method freeze commit.

- [ ] **Step 5: Run final verification before completion**

Run:

```bash
git diff --check
python tools/benchmark_tables.py check --output-dir docs/paper
pytest -q tests/evaluation/test_finalize_tesse_common_v2.py \
  tests/evaluation/test_finalize_tesse_t2.py \
  tests/evaluation/test_import_benchmark_results.py \
  tests/test_benchmark_table_package.py
rg -n '待补充|T2_OVIOVO_|\\bOVIOVO\\b' docs/paper/oviv2_aaai \
  docs/paper/benchmark_tokens.tsv docs/paper/benchmark_tables.md \
  docs/paper/benchmark_tables.tex
git status --short
```

Expected: tests/checks pass, the search emits no active placeholder or legacy method occurrence, and only intended result/paper files remain changed before the final commit.
