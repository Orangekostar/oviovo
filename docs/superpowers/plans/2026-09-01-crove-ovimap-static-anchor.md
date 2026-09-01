# CROVE OVI-MAP Static Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a causal TESSE-CD Apartment method that composes an immutable pre-intervention OVI-MAP static map with CROVE A6 temporal state.

**Architecture:** A one-time adapter converts hash-bound native OVI-MAP outputs from frames 0-262 into the repository's pickle-free neutral snapshot format. A sequential overlay reads CROVE temporal exports and checkpoints in timestamp order, binds temporal identities to anchor entities using pre-intervention centroids, and emits anchor geometry for unchanged/occluded objects while replacing only CROVE-confirmed moved or removed objects. This path is T2-only and does not modify any T1 protected source.

**Tech Stack:** Python 3.10+, NumPy, SciPy, plyfile, pytest, existing OVI-MAP native environments, existing CROVE temporal checkpoints and TESSE-CD evaluators.

---

## File Map

- Create `src/oviv2/ovimap_static_anchor.py`: immutable anchor package, deterministic identity binding, overlay state, and checkpoint composition.
- Create `scripts/evaluation/build_tesse_ovimap_static_anchor.py`: causal-prefix verification and one-time OVI-MAP output conversion.
- Create `scripts/evaluation/run_crove_ovimap_static_anchor.py`: sequential composition of an existing CROVE A6 capture with the anchor package.
- Create `scripts/evaluation/evaluate_crove_ovimap_static_anchor.py`: source-bound Apartment metric execution and hard-gate decision.
- Create `tests/oviv2/test_ovimap_static_anchor.py`: core contract, binding, motion, removal, occlusion, and determinism tests.
- Create `tests/evaluation/test_build_tesse_ovimap_static_anchor.py`: causal input and artifact binding tests.
- Create `tests/evaluation/test_run_crove_ovimap_static_anchor.py`: sequential artifact and publication tests.
- Create `tests/evaluation/test_evaluate_crove_ovimap_static_anchor.py`: metric gate tests.
- Create `configs/evaluation/ovimap_static_anchor_apartment_v1.json`: frozen Apartment thresholds and causal cutoff.
- Create `docs/superpowers/reports/2026-09-01-crove-ovimap-static-anchor-apartment.md` only after a real measured run.

The plan must not modify paths listed in
`configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json`.

### Task 1: Freeze The Causal Prefix Contract

**Files:**
- Create: `configs/evaluation/ovimap_static_anchor_apartment_v1.json`
- Test: `tests/evaluation/test_build_tesse_ovimap_static_anchor.py`
- Create: `scripts/evaluation/build_tesse_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing tests for the exact Apartment prefix**

Add tests that construct a synthetic schedule and RGB-D inventory and require
`load_causal_prefix_contract()` to return frames `0..262`. Also assert rejection
when a consumed frame equals the first intervention, the schedule scene differs,
or the inventory has a gap.

```python
def test_prefix_stops_before_first_intervention(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, scene="apartment", interventions=[263, 711])
    root = _write_rgbd(tmp_path, frame_count=300)
    contract = load_causal_prefix_contract(
        scene="apartment",
        schedule_path=schedule,
        rgbd_root=root,
        configured_cutoff=262,
    )
    assert contract.first_intervention_frame == 263
    assert contract.frame_ids == tuple(range(263))
    assert contract.maximum_source_frame == 262


def test_prefix_rejects_intervention_frame(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path, scene="apartment", interventions=[263])
    root = _write_rgbd(tmp_path, frame_count=264)
    with pytest.raises(ValueError, match="strictly before first intervention"):
        load_causal_prefix_contract(
            scene="apartment",
            schedule_path=schedule,
            rgbd_root=root,
            configured_cutoff=263,
        )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/evaluation/test_build_tesse_ovimap_static_anchor.py
```

Expected: collection fails because
`scripts.evaluation.build_tesse_ovimap_static_anchor` does not exist.

- [ ] **Step 3: Add the frozen configuration**

Create this exact JSON object, then compute and record input hashes at runtime
rather than embedding machine-dependent asset hashes:

```json
{
  "schema_version": 1,
  "method_id": "crove_ovimap_static_anchor_v1",
  "dataset": "TESSE-CD",
  "scene": "apartment",
  "source_role": "causal_pre_intervention_initialization",
  "first_source_frame": 0,
  "last_source_frame": 262,
  "source_stride": 1,
  "ovimap_commit": "58a804e2d7c82ba05a489eb071aba3367301fed8",
  "minimum_spatial_iou": 0.01,
  "maximum_centroid_distance_m": 0.75,
  "minimum_semantic_cosine": 0.65,
  "moved_displacement_m": 0.20,
  "background_voxel_size_m": 0.05
}
```

- [ ] **Step 4: Implement the strict prefix loader**

Define the public contract and loader. The loader must parse JSON structurally,
derive the minimum `intervention_frame_index`, require the configured cutoff to
equal `first_intervention - 1`, and require matching RGB/depth files for every
frame.

Define immutable `CausalPrefixContract(scene, first_intervention_frame,
frame_ids, maximum_source_frame, schedule_sha256,
rgbd_export_manifest_sha256)` and the exact public function
`load_causal_prefix_contract(*, scene: str, schedule_path: Path,
rgbd_root: Path, configured_cutoff: int) -> CausalPrefixContract`.

The implementation must reject symlinks, non-regular files, duplicate schedule
events, non-contiguous frames, and any source frame at or after the intervention.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python -m pytest -q tests/evaluation/test_build_tesse_ovimap_static_anchor.py
python -m py_compile scripts/evaluation/build_tesse_ovimap_static_anchor.py
```

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit**

```bash
git add configs/evaluation/ovimap_static_anchor_apartment_v1.json \
  scripts/evaluation/build_tesse_ovimap_static_anchor.py \
  tests/evaluation/test_build_tesse_ovimap_static_anchor.py
git commit -m "feat: freeze causal OVI-MAP anchor prefix"
```

### Task 2: Run Native OVI-MAP On TESSE Layout

**Files:**
- Modify: `scripts/evaluation/build_tesse_ovimap_static_anchor.py`
- Modify: `tests/evaluation/test_build_tesse_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing native-command tests**

Create a synthetic TESSE RGB-D root containing `cam_params.json`, scene
`traj.txt`, and three RGB/depth pairs. Assert exact CropFormer, geometry, and
mapping argv; assert the mapper receives `--scene_num apartment`, the resolved
RGB-D root after `--data_folder`, `--start 0`, `--end 3`, and `--step 1`. Add
rejections for a missing trajectory, a missing depth frame, a non-pinned OVI-MAP
checkout, and output paths inside the source RGB-D tree.

```python
def test_native_commands_use_exact_causal_tesse_frames(tmp_path: Path) -> None:
    rgbd = _write_rgbd(tmp_path, frame_count=3)
    native = _write_native_environment(tmp_path, commit=PINNED_OVIMAP_COMMIT)
    commands = build_tesse_native_commands(
        scene="apartment",
        rgbd_root=rgbd,
        frame_ids=(0, 1, 2),
        native=native,
        attempt_root=tmp_path / "attempt",
    )
    assert commands.mapping[commands.mapping.index("--scene_num") + 1] == "apartment"
    assert commands.mapping[commands.mapping.index("--end") + 1] == "3"
    assert commands.frame_ids == (0, 1, 2)
```

- [ ] **Step 2: Verify RED**

Run the focused file and confirm `build_tesse_native_commands` is missing.

- [ ] **Step 3: Implement TESSE command construction and preflight**

Define immutable `TesseNativeEnvironment` and `TesseNativeCommands`. Reuse the
existing internal geometry entrypoint from `run_ovimap_native.py`, but build a
new mapping command without its Replica scene whitelist. Require contiguous
frame IDs beginning at zero because upstream OVI-MAP accepts only start/end/step
ranges. Run `git -C "$OVIMAP_ROOT" rev-parse HEAD` and require the exact pinned
commit before command execution. Bind `cam_params.json`, `traj.txt`, every
RGB/depth frame, CropFormer config/weights, SigLIP model, mapper source, compiled
extensions, and environment source hashes in preflight JSON.

- [ ] **Step 4: Implement fail-closed execution**

Expose `run_tesse_native_mapping(*, commands: TesseNativeCommands,
environment: TesseNativeEnvironment) -> Path`. Execute geometry, CropFormer,
and mapping in that order with the existing environment construction rules.
Write each command and exit code before advancing. Require exactly one mask per
source frame plus non-empty final instance mesh, semantic feature pickle, and
instance-color log. Publish `native_mapping_manifest.json` only after all
outputs are hashed and source inventories are revalidated.

- [ ] **Step 5: Verify GREEN**

```bash
python -m pytest -q tests/evaluation/test_build_tesse_ovimap_static_anchor.py
python -m py_compile scripts/evaluation/build_tesse_ovimap_static_anchor.py
```

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluation/build_tesse_ovimap_static_anchor.py \
  tests/evaluation/test_build_tesse_ovimap_static_anchor.py
git commit -m "feat: run OVI-MAP on causal TESSE prefixes"
```

### Task 3: Build A Pickle-Free Anchor Package

**Files:**
- Modify: `scripts/evaluation/build_tesse_ovimap_static_anchor.py`
- Test: `tests/evaluation/test_build_tesse_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing adapter tests**

Use a tiny PLY, feature pickle, and authoritative instance-color log. Require
the builder to preserve all mesh-backed instances, attach available semantic
features, write neutral NPZ/JSONL through `write_map_snapshot`, and write an
exact manifest whose hashes match the outputs. Add a malicious pickle test that
proves runtime loading never opens the source pickle.

```python
def test_build_anchor_binds_mesh_instances_and_hashes_outputs(tmp_path: Path) -> None:
    inputs = _write_native_ovimap_fixture(tmp_path)
    result = build_anchor_package(
        scene="apartment",
        cutoff_frame=2,
        native_manifest=inputs.native_manifest,
        instance_mesh=inputs.instance_mesh,
        semantic_features=inputs.semantic_features,
        instance_color_log=inputs.instance_color_log,
        output_root=tmp_path / "anchor",
    )
    manifest = json.loads(result.manifest.read_text())
    snapshot = read_map_snapshot(result.snapshot, result.entities)
    assert [entity.entity_id for entity in snapshot.entities] == ["ovimap:1", "ovimap:2"]
    assert manifest["causality"]["maximum_source_frame"] == 2
    assert manifest["outputs"]["snapshot"]["sha256"] == _sha256(result.snapshot)
```

- [ ] **Step 2: Verify RED**

Run the focused test and confirm failure because `build_anchor_package` is
missing.

- [ ] **Step 3: Implement conversion and manifest publication**

Reuse `parse_instance_color_log`, `load_instance_mesh`, `bind_mesh_instances`,
and `adapt_ovimap`. Convert OVI-MAP data exactly once, then publish only neutral
NPZ/JSONL and JSON:

Define immutable `AnchorPackagePaths(manifest, snapshot, entities)` and the
exact public function `build_anchor_package(*, scene: str, cutoff_frame: int,
native_manifest: Path, instance_mesh: Path, semantic_features: Path,
instance_color_log: Path, vocabulary_json: Path, siglip_model: Path,
device: str, output_root: Path) -> AnchorPackagePaths`.

Classify eligible instance features with OVI-MAP's existing
`relative_similarity_labels` function, the TESSE vocabulary, the pinned local
SigLIP model, and canonical prompts `object`, `things`, `stuff`, and `texture`.
Set `method="OVI-MAP causal static anchor"`, `scope="current"`,
`first_seen=0`, `last_seen=cutoff_frame`, and add
`anchor_instance_id`, `observation_count`, `authority="ovimap_anchor"`, and
`native_manifest_sha256` to every entity. Manifest publication must use a new
temporary directory, fsync files, and rename only after all hashes are stable.

- [ ] **Step 4: Verify GREEN**

```bash
python -m pytest -q tests/evaluation/test_build_tesse_ovimap_static_anchor.py
```

Expected: prefix and package tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/build_tesse_ovimap_static_anchor.py \
  tests/evaluation/test_build_tesse_ovimap_static_anchor.py
git commit -m "feat: package immutable OVI-MAP anchors"
```

### Task 4: Implement Deterministic Anchor Binding

**Files:**
- Create: `src/oviv2/ovimap_static_anchor.py`
- Create: `tests/oviv2/test_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing one-to-one binding tests**

Cover spatial matches, semantic-only rejection, deterministic tie ordering,
duplicate temporal IDs, and invariance to input ordering. Use prefix samples at
or before cutoff; samples after cutoff must be rejected as binding evidence.

```python
def test_bind_uses_pre_intervention_centroids_and_is_one_to_one() -> None:
    anchor = _anchor_snapshot(ids=("ovimap:1", "ovimap:2"))
    prefix = (
        PrefixIdentitySample(7, 262, (0.0, 0.0, 0.0), "chair", _feature(0)),
        PrefixIdentitySample(8, 262, (2.0, 0.0, 0.0), "table", _feature(1)),
    )
    state = bind_anchor_identities(anchor, prefix, _config(), cutoff_frame=262)
    assert state.bindings == (("ovimap:1", 7), ("ovimap:2", 8))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/oviv2/test_ovimap_static_anchor.py
```

Expected: import failure for `src.oviv2.ovimap_static_anchor`.

- [ ] **Step 3: Implement immutable contracts and Hungarian assignment**

Define:

```python
@dataclass(frozen=True)
class StaticAnchorConfig:
    minimum_spatial_iou: float
    maximum_centroid_distance_m: float
    minimum_semantic_cosine: float
    moved_displacement_m: float
    background_voxel_size_m: float


@dataclass(frozen=True)
class PrefixIdentitySample:
    temporal_entity_id: int
    frame_index: int
    centroid_xyz: tuple[float, float, float]
    semantic_label: str | None
    semantic_embedding: np.ndarray | None


@dataclass(frozen=True)
class AnchorOverlayState:
    cutoff_frame: int
    last_frame_index: int
    bindings: tuple[tuple[str, int], ...]
    initial_geometry_epochs: tuple[tuple[int, int], ...]
    removed_anchor_ids: frozenset[str]
```

Normalize features before cosine similarity, voxelize points at the configured
resolution for IoU, and use `scipy.optimize.linear_sum_assignment` over a
deterministically ordered cost matrix. Invalid pairs receive a finite sentinel
cost and are discarded after assignment. Do not mutate input arrays or
snapshots.

- [ ] **Step 4: Verify GREEN and determinism**

```bash
python -m pytest -q tests/oviv2/test_ovimap_static_anchor.py
python -m pytest -q tests/oviv2/test_ovimap_static_anchor.py --count=2
```

If `pytest-repeat` is unavailable, run the first command twice and compare
output. Expected: identical passing results.

- [ ] **Step 5: Commit**

```bash
git add src/oviv2/ovimap_static_anchor.py tests/oviv2/test_ovimap_static_anchor.py
git commit -m "feat: bind CROVE identities to static anchors"
```

### Task 5: Compose Causal Current Checkpoints

**Files:**
- Modify: `src/oviv2/ovimap_static_anchor.py`
- Modify: `tests/oviv2/test_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing state-transition tests**

Cover all required rules:

- STATIC + ACTIVE emits anchor and suppresses temporal duplicate;
- DYNAMIC + newer geometry epoch + sufficient displacement emits temporal and
  suppresses anchor;
- DORMANT caused by `VISIBLE_ABSENT` suppresses anchor;
- DORMANT caused by `OCCLUDED`, `OUT_OF_VIEW`, or `DEPTH_UNKNOWN` keeps anchor;
- an unbound post-cutoff identity is emitted as NEW;
- a reactivated bound identity retains its original binding;
- checkpoints must increase strictly and cannot precede the cutoff.

```python
def test_occlusion_keeps_anchor_but_visible_absence_removes_it() -> None:
    anchor = _anchor_snapshot(ids=("ovimap:1",))
    bound = _bound_state("ovimap:1", temporal_id=7)
    occluded, state = compose_anchor_checkpoint(
        anchor=anchor,
        temporal=_temporal_checkpoint(entity_id=7, lifecycle="dormant"),
        exports=_exports(entity_id=7, evidence="occluded"),
        state=bound,
        config=_config(),
        anchor_manifest_sha256="a" * 64,
    )
    assert [item.entity_id for item in occluded.entities] == ["ovimap:1"]

    removed, _ = compose_anchor_checkpoint(
        anchor=anchor,
        temporal=_temporal_checkpoint(entity_id=7, lifecycle="dormant", frame=313),
        exports=_exports(entity_id=7, evidence="visible_absent", frame=313),
        state=replace(state, last_frame_index=263),
        config=_config(),
        anchor_manifest_sha256="a" * 64,
    )
    assert removed.entities == []
```

- [ ] **Step 2: Verify RED**

Run the focused test and confirm `compose_anchor_checkpoint` is missing.

- [ ] **Step 3: Implement the causal transition function**

Add:

Define immutable `CheckpointOverlayDiagnostics(frame_index,
unchanged_anchor_ids, moved_anchor_ids, removed_anchor_ids,
new_temporal_ids, occluded_anchor_ids)` and the exact public function
`compose_anchor_checkpoint(*, anchor: MapSnapshot, temporal:
TemporalCurrentSnapshot, exports: tuple[TemporalExportBatch, ...], state:
AnchorOverlayState, config: StaticAnchorConfig, anchor_manifest_sha256: str) ->
tuple[MapSnapshot, AnchorOverlayState, CheckpointOverlayDiagnostics]`.

Use only export batches with `state.last_frame_index < frame_index <=
temporal.metadata.frame_id`. A moved decision requires `DynamicState.DYNAMIC`,
a geometry epoch greater than the bound initial epoch, and anchor/current
centroid displacement at least `moved_displacement_m`. Removal requires a
transition to DORMANT whose evidence is `VISIBLE_ABSENT`. All other absence
evidence preserves the anchor. Merge background point sets by integer voxel key
and deterministic lexicographic order.

- [ ] **Step 4: Verify GREEN**

```bash
python -m pytest -q tests/oviv2/test_ovimap_static_anchor.py
```

Expected: all core binding and composition tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/oviv2/ovimap_static_anchor.py tests/oviv2/test_ovimap_static_anchor.py
git commit -m "feat: compose causal OVI-MAP anchor readouts"
```

### Task 6: Compose An Existing CROVE A6 Capture

**Files:**
- Create: `scripts/evaluation/run_crove_ovimap_static_anchor.py`
- Create: `tests/evaluation/test_run_crove_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing end-to-end fixture tests**

Construct a tiny source run containing a run manifest, temporal export JSONL,
two temporal checkpoints, and an anchor package. Assert chronological output,
complete source hashes, no overwrite of an existing output, and rejection of a
source run whose checkpoint precedes its claimed temporal export.

```python
def test_composed_run_publishes_hash_bound_checkpoints(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    anchor = _write_anchor_package(tmp_path)
    result = compose_run(
        source_run_manifest=source.manifest,
        anchor_manifest=anchor.manifest,
        output_root=tmp_path / "composed",
    )
    manifest = json.loads(result.read_text())
    assert manifest["status"] == "PASS"
    assert manifest["method"] == "CROVE + OVI-MAP static anchor (composed)"
    assert [item["frame_index"] for item in manifest["checkpoints"]] == [263, 313]
```

- [ ] **Step 2: Verify RED**

Run the file and confirm import failure for the new runner.

- [ ] **Step 3: Implement source-bound sequential composition**

Define:

Implement the exact public function `compose_run(*, source_run_manifest: Path,
anchor_manifest: Path, output_root: Path) -> Path`.

The runner must validate all declared paths, hashes, byte counts, scene IDs,
checkpoint frame/timestamp order, and complete temporal export coverage. Derive
prefix identity samples only from exports with frame index at most the cutoff.
Load checkpoints with existing no-pickle loaders, call
`compose_anchor_checkpoint` sequentially, publish with `write_map_snapshot`,
and emit one diagnostics JSON per checkpoint. The final manifest binds every
input and output and records `integration="composed"` and
`execution_mode="online_after_causal_initialization"`.

- [ ] **Step 4: Add CLI**

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    compose_run(
        source_run_manifest=args.source_run_manifest,
        anchor_manifest=args.anchor_manifest,
        output_root=args.output_root,
    )
    return 0
```

- [ ] **Step 5: Verify GREEN**

```bash
python -m pytest -q tests/evaluation/test_run_crove_ovimap_static_anchor.py
python -m py_compile scripts/evaluation/run_crove_ovimap_static_anchor.py
```

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluation/run_crove_ovimap_static_anchor.py \
  tests/evaluation/test_run_crove_ovimap_static_anchor.py
git commit -m "feat: compose CROVE runs with OVI-MAP anchors"
```

### Task 7: Add The Apartment Quality Gate

**Files:**
- Create: `scripts/evaluation/evaluate_crove_ovimap_static_anchor.py`
- Create: `tests/evaluation/test_evaluate_crove_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing threshold tests**

Test exact boundary acceptance and one failure per hard metric. Missing or
non-finite Dyn. F1 must fail. A failure must return `REJECTED_RETAIN_A6` and
must not produce paper token bindings.

```python
def test_apartment_gate_accepts_exact_static_floors_and_dynamic_gains() -> None:
    decision = decide_apartment_gate(
        obj_f1=0.372762,
        dyn_f1=0.01,
        chg_f1=0.060854,
        current_miou=0.142897,
        ghost_rate=0.646882,
        processed_frames=1745,
        official_state_count=43,
    )
    assert decision.status == "PASS_APARTMENT"
    assert decision.office_authorized is True
```

- [ ] **Step 2: Verify RED**

Run the focused file and confirm import failure.

- [ ] **Step 3: Implement exact gate logic and provenance receipt**

Define an immutable `ApartmentGateDecision` and `decide_apartment_gate()` with
the values from the approved design. The CLI invokes the existing official and
common-v2 evaluators, repeats metric aggregation, requires byte-identical
results, then writes a receipt containing source hashes and metric deltas. It
must never update `benchmark_tokens.tsv` or paper tables.

- [ ] **Step 4: Verify GREEN**

```bash
python -m pytest -q tests/evaluation/test_evaluate_crove_ovimap_static_anchor.py
python -m py_compile scripts/evaluation/evaluate_crove_ovimap_static_anchor.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/evaluate_crove_ovimap_static_anchor.py \
  tests/evaluation/test_evaluate_crove_ovimap_static_anchor.py
git commit -m "feat: gate anchored Apartment results"
```

### Task 8: Restore Native OVI-MAP And Produce The Real Apartment Anchor

**Files:**
- External build: `/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/`
- External run: `/home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/`
- Create after measurement: `docs/superpowers/reports/2026-09-01-crove-ovimap-static-anchor-apartment.md`

- [ ] **Step 1: Rebuild the pinned native environment**

Use the tracked bootstrap and verify the source commit without modifying the
checkout:

```bash
bash scripts/reproduction/ovimap/bootstrap_native_envs.sh
git -C /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP \
  rev-parse HEAD
```

Expected commit:
`58a804e2d7c82ba05a489eb071aba3367301fed8`.

- [ ] **Step 2: Run the causal-prefix native mapping**

Execute the TESSE-compatible native command builder from Task 2 with Apartment
frames 0-262, stride 1, the RGB-D root
`/home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1`, and a new immutable
attempt directory under the external run root. Expected outputs include the
instance mesh, semantic feature pickle, authoritative color log, per-frame
inventory, and native manifest.

- [ ] **Step 3: Build the neutral anchor package**

```bash
python scripts/evaluation/build_tesse_ovimap_static_anchor.py \
  --config configs/evaluation/ovimap_static_anchor_apartment_v1.json \
  --schedule configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json \
  --rgbd-root /home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1 \
  --native-manifest /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/native/apartment/native_mapping_manifest.json \
  --output-root /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/anchor/apartment
```

Expected: manifest status `PASS`, `maximum_source_frame=262`, and no consumed
frame above 262.

- [ ] **Step 4: Compose with the source-bound A6 capture**

Locate the A6 run manifest by its recorded SHA-256
`e7a179d0ee5f901875db3cffaacfcbd255d47401a9e703b9bbbd5c0d39a42899`,
verify it before use, then run:

```bash
SOURCE_RUN=$(
  find /home/ww/oviovo_baseline_runs -type f -name 'run_manifest.json' -print0 \
    | xargs -0 sha256sum \
    | awk '$1 == "e7a179d0ee5f901875db3cffaacfcbd255d47401a9e703b9bbbd5c0d39a42899" {print $2; exit}'
)
test -n "$SOURCE_RUN"
python scripts/evaluation/run_crove_ovimap_static_anchor.py \
  --source-run-manifest "$SOURCE_RUN" \
  --anchor-manifest /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/anchor/apartment/anchor_manifest.json \
  --output-root /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/composed/apartment
```

- [ ] **Step 5: Evaluate and apply the gate**

Run the new evaluator twice. Expected: byte-identical metrics and a measured
PASS or REJECT decision. Only PASS authorizes Office.

- [ ] **Step 6: Run regression and source-integrity checks**

```bash
python -m pytest -q \
  tests/oviv2/test_ovimap_static_anchor.py \
  tests/evaluation/test_build_tesse_ovimap_static_anchor.py \
  tests/evaluation/test_run_crove_ovimap_static_anchor.py \
  tests/evaluation/test_evaluate_crove_ovimap_static_anchor.py
python -m pytest -q tests/oviv2/test_t1_noninterference.py tests/oviv2/test_t1_exactness.py
python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py \
  --verify-source-manifest configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json
git diff --check
```

Expected: all tests pass and the protected-source verifier exits 0.

- [ ] **Step 7: Record only measured results**

Create the report with artifact hashes, exact commands, Obj/Dyn/Chg, current
mIoU, ghost rate, background F-score, decision, and failure diagnostics. If the
gate fails, state `REJECTED; RETAIN A6` and do not run Office or update paper
tables.

## Self-Review

- Spec coverage: causal prefix, TESSE-native OVI-MAP execution, immutable anchor,
  sticky identity binding,
  occlusion-safe removal, moved/new readout, deterministic publication,
  Apartment-only gate, and T1 noninterference each have a dedicated task.
- Placeholder scan: execution paths that depend on generated immutable attempt
  names are represented as CLI inputs; no implementation behavior is deferred.
- Type consistency: `CausalPrefixContract`, `AnchorPackagePaths`,
  `StaticAnchorConfig`, `PrefixIdentitySample`, `AnchorOverlayState`,
  `CheckpointOverlayDiagnostics`, and `compose_anchor_checkpoint` are introduced
  once and reused with the same signatures.
