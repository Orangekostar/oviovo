# OVIV2 Stage 2 RADSeg Dense Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a causal, reproducible RADSeg probability branch that projects current-frame dense open-vocabulary predictions into OVIV2's sparse semantic evidence and independently evaluates the dense semantic head.

**Architecture:** Run RADSeg or the RayFronts NARADIO fallback in an isolated Python 3.11 worker and freeze each frame as a checksummed top-k probability cache. The existing OVIV2 runner reads those caches in frame order, integrates them into `SparseEvidenceStore`, persists exact model/cache provenance in snapshot schema 3, and evaluates `dense_only` separately from `owner_authoritative`; Stage 3 fusion remains out of scope.

**Tech Stack:** Python 3.10/3.11, NumPy, PyTorch 2.4, RADSeg, RayFronts, C-RADIOv3, JSONL subprocess protocol, compressed NPZ, Open3D, pytest.

---

## File Structure

- Create `configs/environments/oviv2_radseg.yaml`: isolated inference environment.
- Create `docs/paper/licenses/oviv2_stage2_dense_frontends.json`: pinned source/model license audit.
- Create `src/oviv2/dense_semantics.py`: immutable frame/cache/provenance contracts and cache loader.
- Create `src/oviv2/dense_projection.py`: depth projection, quality weighting, and sparse evidence integration.
- Create `scripts/radseg_dense_worker.py`: persistent RADSeg/NARADIO inference worker.
- Create `scripts/precompute_oviv2_dense_semantics.py`: atomic dense-cache producer.
- Create `configs/oviv2_replica_room0_precision_stage2_radseg.json`: Stage 2 room0 run config.
- Modify `src/oviv2/runtime.py`: optional dense-frame integration.
- Modify `src/oviv2/snapshot.py`: schema 3 dense provenance.
- Modify `src/oviv2/meshing.py`: no behavior change; only use its existing `entity_semantics=None` dense path.
- Modify `scripts/run_oviv2_replica.py`: dense-cache preflight, runtime wiring, dual semantic-head exports.
- Modify `scripts/evaluation/evaluate_oviv2_replica.py`: explicit semantic-head selection.
- Create/modify focused tests under `tests/oviv2` and `tests/evaluation`.

### Task 1: Freeze External Sources, Licenses, And Environment

**Files:**
- Create: `configs/environments/oviv2_radseg.yaml`
- Create: `docs/paper/licenses/oviv2_stage2_dense_frontends.json`
- External read-only clones: `/home/ww/oviovo_references/modules/{RADSeg,RayFronts,RADIO}`

- [ ] **Step 1: Pin the three official repositories without submodules**

```bash
mkdir -p /home/ww/oviovo_references/modules
git clone https://github.com/RADSeg-OVSS/RADSeg.git /home/ww/oviovo_references/modules/RADSeg
git -C /home/ww/oviovo_references/modules/RADSeg checkout --detach 3fe8789a3c1b11e41688f7deadc8b8db088ef1a7
git clone https://github.com/RayFronts/RayFronts.git /home/ww/oviovo_references/modules/RayFronts
git -C /home/ww/oviovo_references/modules/RayFronts checkout --detach 031262a9ed4d0ea456a1d5605df835ba9e53b027
git clone https://github.com/NVlabs/RADIO.git /home/ww/oviovo_references/modules/RADIO
git -C /home/ww/oviovo_references/modules/RADIO checkout --detach c0f37017930e9dda53f93424cf4bf39fc51f287e
```

Expected: all repositories are detached, clean, and no submodule is initialized; never write inside these clones.

- [ ] **Step 2: Write the isolated environment definition**

```yaml
name: oviovo-radseg
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.4.0
  - torchvision=0.19.0
  - pytorch-cuda=12.1
  - pip
  - pip:
      - timm==1.0.19
      - segment-anything
      - pillow
      - numpy
      - scipy
      - scikit-image
      - transformers
      - typing_extensions
      - einops
```

- [ ] **Step 3: Write the machine-readable license audit**

Use this exact content:

```json
{
  "schema_version": 1,
  "radseg": {
    "commit": "3fe8789a3c1b11e41688f7deadc8b8db088ef1a7",
    "license": "MIT",
    "upstream": "https://github.com/RADSeg-OVSS/RADSeg"
  },
  "rayfronts": {
    "commit": "031262a9ed4d0ea456a1d5605df835ba9e53b027",
    "license": "MIT",
    "upstream": "https://github.com/RayFronts/RayFronts"
  },
  "radio": {
    "commit": "c0f37017930e9dda53f93424cf4bf39fc51f287e",
    "code_license": "NVIDIA Source Code License-NC",
    "c_radio_v3_model_license": "NVIDIA Open Model License",
    "radio_v2_5_model_license": "NVIDIA Source Code License-NC",
    "upstream": "https://github.com/NVlabs/RADIO"
  },
  "sam_vit_h": {
    "path": "/home/ww/oviovo_benchmark_assets/weights/segment_anything/sam_vit_h_4b8939.pth",
    "sha256": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    "license": "Apache-2.0"
  }
}
```

- [ ] **Step 4: Create and verify the environment**

```bash
conda env create -f configs/environments/oviv2_radseg.yaml
/home/ww/miniconda3/envs/oviovo-radseg/bin/python -c 'import torch, timm, segment_anything; print(torch.__version__, torch.cuda.is_available())'
git -C /home/ww/oviovo_references/modules/RADSeg status --short
git -C /home/ww/oviovo_references/modules/RayFronts status --short
git -C /home/ww/oviovo_references/modules/RADIO status --short
```

Expected: PyTorch `2.4.0`, CUDA `True`, and three empty status outputs.

- [ ] **Step 5: Commit environment and audit only**

```bash
git add configs/environments/oviv2_radseg.yaml docs/paper/licenses/oviv2_stage2_dense_frontends.json
git commit -m "build: pin OVIV2 dense semantic frontends"
```

### Task 2: Define Immutable Dense Cache Contracts

**Files:**
- Create: `src/oviv2/dense_semantics.py`
- Create: `tests/oviv2/test_dense_semantics.py`

- [ ] **Step 1: Write failing validation and round-trip tests**

```python
def test_dense_frame_requires_aligned_topk_arrays() -> None:
    with pytest.raises(ValueError, match="same shape"):
        DenseSemanticFrame(
            cache_frame_id=0,
            source_frame_id=0,
            image_shape=(4, 6),
            sample_stride=2,
            class_count=3,
            class_ids=np.ones((2, 3, 2), dtype=np.int64),
            probabilities=np.ones((2, 3, 1), dtype=np.float32),
            entropy=np.zeros((2, 3), dtype=np.float32),
            margin=np.ones((2, 3), dtype=np.float32),
        )


def test_dense_frame_arrays_cannot_be_made_writeable() -> None:
    frame = dense_frame()
    for value in (frame.class_ids, frame.probabilities, frame.entropy, frame.margin):
        with pytest.raises(ValueError):
            value.flags.writeable = True


def test_dense_cache_round_trip_rejects_wrong_hash(tmp_path: Path) -> None:
    write_dense_frame(tmp_path / "frame000000.npz", dense_frame())
    digest = sha256_file(tmp_path / "frame000000.npz")
    loaded = load_dense_frame(tmp_path / "frame000000.npz", expected_sha256=digest)
    expected = dense_frame()
    assert loaded.cache_frame_id == expected.cache_frame_id
    assert loaded.source_frame_id == expected.source_frame_id
    assert np.array_equal(loaded.class_ids, expected.class_ids)
    assert np.allclose(loaded.probabilities, expected.probabilities)
    with pytest.raises(ValueError, match="checksum"):
        load_dense_frame(tmp_path / "frame000000.npz", expected_sha256="0" * 64)
```

- [ ] **Step 2: Run the tests and confirm failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/oviv2/test_dense_semantics.py
```

Expected: collection fails because `src.oviv2.dense_semantics` does not exist.

- [ ] **Step 3: Implement the contracts and atomic NPZ codec**

Implement these public values:

```python
@dataclass(frozen=True)
class DenseSemanticProvenance:
    backend: str
    source_commit: str
    radio_commit: str
    model_id: str
    model_sha256: str
    auxiliary_model_sha256: str
    vocabulary_sha256: str
    prompt_sha256: str
    inference_config_sha256: str
    cache_prefix_sha256: str


@dataclass(frozen=True, eq=False)
class DenseSemanticFrame:
    cache_frame_id: int
    source_frame_id: int
    image_shape: tuple[int, int]
    sample_stride: int
    class_count: int
    class_ids: np.ndarray
    probabilities: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.class_ids, dtype=np.int64)
        probabilities = np.asarray(self.probabilities, dtype=np.float32)
        entropy = np.asarray(self.entropy, dtype=np.float32)
        margin = np.asarray(self.margin, dtype=np.float32)
        if ids.ndim != 3 or probabilities.shape != ids.shape:
            raise ValueError("class_ids and probabilities must have the same shape [H,W,K]")
        if entropy.shape != ids.shape[:2] or margin.shape != ids.shape[:2]:
            raise ValueError("entropy and margin must match the sampled image shape")
        if np.any(ids < 0) or np.any(ids > self.class_count):
            raise ValueError("class_ids lie outside the frozen vocabulary")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("probabilities must be finite and lie in [0, 1]")
        for name, value in (("entropy", entropy), ("margin", margin)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "class_ids", immutable_array(ids))
        object.__setattr__(self, "probabilities", immutable_array(probabilities))
        object.__setattr__(self, "entropy", immutable_array(entropy))
        object.__setattr__(self, "margin", immutable_array(margin))
```

`write_dense_frame()` must use a temporary file, `np.savez_compressed`, `fsync`, and `os.replace`. `load_dense_frame()` must use `allow_pickle=False`, validate schema version `1`, verify the SHA256 before opening, and return a fully validated `DenseSemanticFrame`.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/oviv2/test_dense_semantics.py
git add src/oviv2/dense_semantics.py tests/oviv2/test_dense_semantics.py
git commit -m "feat: add OVIV2 dense semantic cache contracts"
```

Expected: all focused tests pass.

### Task 3: Implement The Persistent RADSeg/NARADIO Worker

**Files:**
- Create: `scripts/radseg_dense_worker.py`
- Create: `tests/oviv2/test_radseg_dense_worker.py`

- [ ] **Step 1: Write protocol and probability-reduction tests**

```python
def test_reduce_probabilities_emits_one_based_topk_and_uncertainty() -> None:
    probabilities = np.asarray([[[[0.1, 0.7], [0.4, 0.2]], [[0.9, 0.3], [0.6, 0.8]]]])
    reduced = reduce_probabilities(probabilities, sample_stride=1, top_k=2)
    assert reduced["class_ids"].tolist() == [[[2, 1], [1, 2]], [[2, 1], [2, 1]]]
    assert np.all(reduced["probabilities"][..., 0] >= reduced["probabilities"][..., 1])
    assert np.all((0.0 <= reduced["margin"]) & (reduced["margin"] <= 1.0))
    assert np.all(reduced["entropy"] >= 0.0)


def test_worker_request_rejects_wrong_rgb_byte_count() -> None:
    with pytest.raises(ValueError, match="RGB"):
        decode_rgb({"shape": [2, 2, 3], "dtype": "uint8", "encoding": "base64", "data": "AA=="})


def test_metadata_response_contains_all_provenance_hashes(fake_backend) -> None:
    response = run_request(fake_backend, {"id": 1, "operation": "metadata"})
    assert response["ok"] is True
    assert set(response["provenance"]) >= {
        "backend", "source_commit", "radio_commit", "model_id", "model_sha256",
        "vocabulary_sha256", "prompt_sha256", "inference_config_sha256",
    }
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/oviv2/test_radseg_dense_worker.py
```

Expected: collection fails because the worker module does not exist.

- [ ] **Step 3: Implement a common backend protocol**

The worker CLI must require pinned local source paths and never invoke an unpinned remote repository:

```python
parser.add_argument("--backend", choices=("radseg", "naradio"), required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--radio-root", type=Path, required=True)
parser.add_argument("--model-version", required=True)
parser.add_argument("--lang-model", required=True)
parser.add_argument("--classes-json", type=Path, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--sample-stride", type=int, default=4)
parser.add_argument("--top-k", type=int, default=4)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--sam-refinement", action="store_true")
parser.add_argument("--sam-checkpoint", type=Path)
```

During model construction, temporarily route calls whose repository argument equals `NVlabs/RADIO` to the pinned local RADIO checkout with `source="local"`; pass every original keyword unchanged and restore `torch.hub.load` immediately afterward. RADSeg uses `predict=True`, the frozen 41-class list, `scra_scaling=10`, `scga_scaling=10`, `slide_crop=336`, `slide_stride=112`, and `prompt_denoising_thresh=0.5`. NARADIO computes language-aligned features and applies `softmax(100 * cosine)` over the same class list.

- [ ] **Step 4: Implement deterministic sampled top-k responses**

```python
def reduce_probabilities(probabilities: np.ndarray, sample_stride: int, top_k: int) -> dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("probabilities must have shape [1,C,H,W]")
    sampled = values[0, :, ::sample_stride, ::sample_stride].transpose(1, 2, 0)
    order = np.argsort(-sampled, axis=-1, kind="stable")[..., :top_k]
    top = np.take_along_axis(sampled, order, axis=-1)
    mass = sampled.sum(axis=-1, keepdims=True)
    normalized = np.divide(sampled, mass, out=np.zeros_like(sampled), where=mass > 0.0)
    entropy = -np.sum(np.where(normalized > 0.0, normalized * np.log(normalized), 0.0), axis=-1)
    margin = top[..., 0] - (top[..., 1] if top.shape[-1] > 1 else 0.0)
    return {
        "class_ids": order.astype(np.int64) + 1,
        "probabilities": top.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "margin": margin.astype(np.float32),
    }
```

Return arrays as base64 blocks with explicit dtype and shape. Every request returns exactly one JSON line; model errors use `{"id": request_id, "ok": false, "error": str(exc)}` without terminating the worker.

- [ ] **Step 5: Hash loaded source, prompts, config, and checkpoint**

Fail startup unless both source roots are clean detached checkouts. After model load, identify the single checkpoint used under `torch.hub.get_dir()/checkpoints`, hash it, and expose the hash in the metadata response. Set `auxiliary_model_sha256` to the SAM checkpoint hash for RADSeg+ and to the empty string for other backends. The model ID format is:

```python
model_id = (
    f"radseg:{args.model_version}:{args.lang_model}:"
    f"sam={int(args.sam_refinement)}:sha256={checkpoint_sha256}"
    if args.backend == "radseg"
    else f"naradio:{args.model_version}:{args.lang_model}:sha256={checkpoint_sha256}"
)
```

- [ ] **Step 6: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/oviv2/test_radseg_dense_worker.py
git add scripts/radseg_dense_worker.py tests/oviv2/test_radseg_dense_worker.py
git commit -m "feat: add pinned RADSeg dense inference worker"
```

### Task 4: Build Atomic Dense Probability Caches

**Files:**
- Create: `scripts/precompute_oviv2_dense_semantics.py`
- Create: `tests/evaluation/test_precompute_oviv2_dense_semantics.py`

- [ ] **Step 1: Write failing CLI tests with a fake JSONL worker**

```python
def test_precompute_writes_atomic_manifest_and_checksums(tmp_path: Path) -> None:
    output = tmp_path / "dense"
    run_precompute(output=output, num_frames=2, worker_script=fake_worker)
    manifest = json.loads((output / "dense_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["frame_count"] == 2
    assert sorted(manifest["cache_files_sha256"]) == ["frame000000.npz", "frame000001.npz"]
    for name, digest in manifest["cache_files_sha256"].items():
        assert sha256_file(output / name) == digest


def test_precompute_refuses_existing_output_without_verified_resume(tmp_path: Path) -> None:
    output = tmp_path / "dense"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_precompute(output=output, num_frames=1, worker_script=fake_worker)
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/evaluation/test_precompute_oviv2_dense_semantics.py
```

- [ ] **Step 3: Implement the producer**

The producer loads scene/frame selection from the existing OVIV2 JSON config, launches one persistent `JsonLineWorkerClient`, verifies the metadata response before creating the output directory, sends RGB frames in cache order, reconstructs arrays, writes each frame with `write_dense_frame`, and writes `dense_manifest.json` last through `_atomic_json`.

The manifest must contain:

```python
manifest = {
    "schema_version": 1,
    "method": "OVIV2-dense-semantic-cache",
    "scene": config["scene"],
    "frame_count": num_frames,
    "source_frame_ids": source_ids,
    "image_shape": list(frame_shape),
    "sample_stride": sample_stride,
    "top_k": top_k,
    "class_count": len(classes),
    "vocabulary_sha256": vocabulary_hash,
    "provenance": metadata["provenance"],
    "cache_files_sha256": cache_hashes,
}
```

Resume is allowed only when the completed manifest covers the requested prefix and every recorded file hash matches; no partial directory is treated as complete.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/evaluation/test_precompute_oviv2_dense_semantics.py
git add scripts/precompute_oviv2_dense_semantics.py tests/evaluation/test_precompute_oviv2_dense_semantics.py
git commit -m "feat: precompute immutable OVIV2 dense semantics"
```

### Task 5: Project Dense Probabilities Into Sparse Evidence

**Files:**
- Create: `src/oviv2/dense_projection.py`
- Create: `tests/oviv2/test_dense_projection.py`

- [ ] **Step 1: Write failing causal projection tests**

```python
def test_integrator_projects_topk_probabilities_to_expected_voxel() -> None:
    frame = make_frame(depth=np.full((4, 4), 2.0, dtype=np.float32))
    dense = make_dense_frame(class_ids=[[[2, 3]]], probabilities=[[[0.8, 0.2]]], stride=4)
    store = SparseEvidenceStore(EvidenceConfig(block_resolution=8, semantic_top_k=4))
    result = DenseSemanticIntegrator(DenseSemanticConfig(voxel_size_m=0.5)).integrate(
        frame, dense, store, revision=1
    )
    assert result.updated_voxel_count == 1
    candidates = store.semantic_candidates((0, 0, 4))
    assert [item.label_id for item in candidates] == [2, 3]
    assert candidates[0].support > candidates[1].support > 0.0


def test_integrator_drops_invalid_depth_high_entropy_and_out_of_radius() -> None:
    frame = make_frame(depth=np.asarray([[np.nan, 1.0, 9.0]], dtype=np.float32))
    dense = make_dense_frame_for_row(entropy=[0.0, np.log(3), 0.0], stride=1)
    result = DenseSemanticIntegrator(
        DenseSemanticConfig(voxel_size_m=0.5, integration_radius_m=3.0, minimum_quality=0.01)
    ).integrate(frame, dense, store, revision=1)
    assert result.updated_voxel_count == 0


def test_integrator_never_reads_future_frame() -> None:
    frame = make_frame(frame_id=10)
    dense = make_dense_frame(source_frame_id=20)
    with pytest.raises(ValueError, match="frame"):
        integrator.integrate(frame, dense, store, revision=1)
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/oviv2/test_dense_projection.py
```

- [ ] **Step 3: Implement bounded quality weighting and vectorized projection**

```python
@dataclass(frozen=True)
class DenseSemanticConfig:
    voxel_size_m: float
    integration_radius_m: float = 6.0
    minimum_probability: float = 0.01
    minimum_quality: float = 0.01
    entropy_power: float = 1.0
    view_angle_power: float = 1.0


quality = (
    np.clip(1.0 - entropy / np.log(dense.class_count), 0.0, 1.0) ** config.entropy_power
    * np.clip(view_cosine, 0.0, 1.0) ** config.view_angle_power
)
support = quality[..., None] * dense.probabilities
```

Estimate camera-space normals from adjacent valid depth samples with cross products; use a view cosine of `1.0` only when a normal cannot be estimated. Back-project sampled rows/columns with the frame intrinsics and pose, reject nonfinite/nonpositive depth and points beyond `integration_radius_m`, quantize with `floor(world / voxel_size_m)`, aggregate duplicate `(voxel, class)` support using `np.unique`, then call `update_semantic` in sorted order. Return counts for sampled pixels, valid pixels, and updated voxels.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/oviv2/test_dense_projection.py
git add src/oviv2/dense_projection.py tests/oviv2/test_dense_projection.py
git commit -m "feat: integrate dense semantics into sparse evidence"
```

### Task 6: Persist Dense Provenance In Snapshot Schema 3

**Files:**
- Modify: `src/oviv2/snapshot.py`
- Modify: `src/oviv2/runtime.py`
- Modify: `tests/oviv2/test_snapshot.py`
- Modify: `tests/oviv2/test_runtime.py`

- [ ] **Step 1: Write failing schema and runtime tests**

```python
def test_schema3_round_trip_preserves_dense_provenance(tmp_path: Path) -> None:
    metadata = VoxelSnapshotMetadata(
        scene_id="room0", frame_id=1, timestamp=1.0, revision=1,
        voxel_size_m=0.05, block_resolution=8, schema_version=3,
        dense_semantic_provenance=provenance(),
    )
    loaded = commit_snapshot(tmp_path, metadata)
    assert loaded.metadata.dense_semantic_provenance == provenance()


def test_schema3_requires_dense_provenance_and_registry() -> None:
    with pytest.raises(ValueError, match="dense semantic provenance"):
        VoxelSnapshotMetadata(
            scene_id="room0",
            frame_id=1,
            timestamp=1.0,
            revision=1,
            voxel_size_m=0.05,
            block_resolution=8,
            schema_version=3,
            dense_semantic_provenance=None,
        )


def test_runtime_integrates_dense_frame_transactionally() -> None:
    runtime = Oviv2Runtime("room0", config_with_dense(), dense_semantic_provenance=provenance())
    runtime.process_frame(frame, observations=(), dense_semantics=dense_frame())
    assert runtime.evidence.semantic_candidates(expected_voxel)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_snapshot.py tests/oviv2/test_runtime.py
```

- [ ] **Step 3: Extend metadata without changing v1/v2 behavior**

Add `dense_semantic_provenance: DenseSemanticProvenance | None = None`. Schema 1 and 2 reject non-`None` provenance; schema 3 requires it and a registry. `_data_files(3)` remains the schema-2 file set because provenance is checksummed inside `metadata.json`. In `VoxelSnapshotMetadata.__post_init__`, convert a JSON dictionary to `DenseSemanticProvenance(**value)` before validation so schema-3 loading reconstructs the typed object; loading schema 1/2 must produce `None`.

- [ ] **Step 4: Integrate dense evidence transactionally**

Add `dense_semantics: DenseSemanticConfig | None` to `Oviv2RuntimeConfig`, construct `DenseSemanticIntegrator` only when enabled, and extend:

```python
def process_frame(
    self,
    frame: Frame,
    observations: tuple[FrameObservation, ...],
    dense_semantics: DenseSemanticFrame | None = None,
) -> RuntimeFrameResult:
```

Apply dense updates to a trial copy of `SparseEvidenceStore` before publishing tracker/registry/evidence state. Any dense validation failure leaves revision, tracker, registry, evidence, and ownership unchanged. Add dense sample/update counters to `RuntimeFrameResult`.

- [ ] **Step 5: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2/test_snapshot.py tests/oviv2/test_runtime.py tests/oviv2/test_dense_projection.py
git add src/oviv2/snapshot.py src/oviv2/runtime.py tests/oviv2/test_snapshot.py tests/oviv2/test_runtime.py
git commit -m "feat: persist OVIV2 dense semantic state"
```

### Task 7: Add Explicit Dense-Only Evaluation

**Files:**
- Modify: `scripts/evaluation/evaluate_oviv2_replica.py`
- Modify: `tests/evaluation/test_evaluate_oviv2_replica_cli.py`
- Modify: `tests/evaluation/test_oviv2_replica.py`

- [ ] **Step 1: Write failing semantic-head tests**

```python
def test_dense_only_head_uses_voxel_semantics_but_preserves_entity_ids(snapshot3, args) -> None:
    args.semantic_head = "dense_only"
    metrics = evaluate(args)
    projected_semantics = np.load(args.output / "gt_aligned_semantic_ids.npy")
    projected_instances = np.load(args.output / "gt_aligned_instance_ids.npy")
    assert metrics["protocol"]["semantic_head"] == "dense_only"
    assert np.any(projected_semantics == DENSE_LABEL)
    assert np.any(projected_instances > 0)


def test_unknown_semantic_head_is_rejected() -> None:
    with pytest.raises(ValueError, match="semantic_head"):
        evaluate(args_with_head("fused"))
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/evaluation/test_oviv2_replica.py
```

- [ ] **Step 3: Implement the head switch**

Accept only `owner_authoritative` and `dense_only`. Always load entity records for instance evaluation. Call:

```python
mesh = derive_labeled_mesh(
    snapshot.geometry,
    snapshot.evidence,
    snapshot.ownership,
    entity_semantics=(entity_semantics if semantic_head == "owner_authoritative" else None),
)
metrics["protocol"]["semantic_head"] = semantic_head
```

Add `--semantic-head` with default `owner_authoritative`; include the head in every output protocol so directories cannot be confused.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/evaluation/test_oviv2_replica.py
git add scripts/evaluation/evaluate_oviv2_replica.py \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/evaluation/test_oviv2_replica.py
git commit -m "feat: evaluate OVIV2 dense semantic head"
```

### Task 8: Wire Dense Caches Into The Replica Runner

**Files:**
- Modify: `scripts/run_oviv2_replica.py`
- Modify: `tests/evaluation/test_run_oviv2_replica_cli.py`
- Create: `configs/oviv2_replica_room0_precision_stage2_radseg.json`

- [ ] **Step 1: Write failing preflight, manifest, and dual-output tests**

```python
def test_dense_mode_rejects_missing_or_mismatched_manifest(tmp_path: Path) -> None:
    config = stage1_config(dense_semantic_mode="cached_probabilities", dense_cache_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="dense_manifest"):
        run(config)


def test_stage2_run_records_dense_provenance_and_two_evaluations(tmp_path: Path) -> None:
    manifest = run_stage2_fixture(tmp_path)
    assert manifest["dense_semantics"]["mode"] == "cached_probabilities"
    assert manifest["dense_semantics"]["provenance"]["model_sha256"] == MODEL_HASH
    assert (tmp_path / "evaluation_owner" / "metrics.json").is_file()
    assert (tmp_path / "evaluation_dense" / "metrics.json").is_file()
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q tests/evaluation/test_run_oviv2_replica_cli.py
```

- [ ] **Step 3: Add strict dense-cache preflight**

When `dense_semantic_mode == "cached_probabilities"`, require `dense_cache_dir`, load `dense_manifest.json`, verify scene, vocabulary hash, source frame IDs, class count, sample stride, top-k, requested prefix coverage, every frame checksum, and every provenance field. Compute a prefix hash by hashing `(cache_index, frame_sha256)` in order and construct `DenseSemanticProvenance` from the manifest. Disabled mode must not read or require a dense cache.

- [ ] **Step 4: Wire cache frames and export both heads**

For each frame, load exactly `frame{cache_index:06d}.npz`, require matching cache/source IDs, and pass it to `runtime.process_frame`. Record dense counters per frame. Keep `semantic_mode="owner_authoritative"` for the primary entity path and write:

```text
final/oviv2_owner_mesh.ply
final/oviv2_dense_mesh.ply
evaluation_owner/metrics.json
evaluation_dense/metrics.json
```

During Stage 2 development, `evaluation/` must be absent so no head is silently treated as the final benchmark result.

- [ ] **Step 5: Create the Stage 2 config**

Copy all Stage 1 values and add only:

```json
{
  "dense_semantic_mode": "cached_probabilities",
  "dense_cache_dir": "/home/ww/oviovo_dense_cache/room0_radseg_l_sam_s4_k4",
  "dense_integration_radius_m": 6.0,
  "dense_minimum_probability": 0.01,
  "dense_minimum_quality": 0.01,
  "dense_entropy_power": 1.0,
  "dense_view_angle_power": 1.0
}
```

- [ ] **Step 6: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_run_oviv2_replica_cli.py tests/oviv2/test_runtime.py
git add scripts/run_oviv2_replica.py tests/evaluation/test_run_oviv2_replica_cli.py \
  configs/oviv2_replica_room0_precision_stage2_radseg.json
git commit -m "feat: run OVIV2 with frozen dense semantics"
```

### Task 9: Run Real Two-Frame Model And Mapping Gates

**Files:**
- Generated outputs only; do not commit caches or run directories.

- [ ] **Step 1: Smoke RADSeg-base on GPU 0**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ww/miniconda3/envs/oviovo-radseg/bin/python \
  scripts/precompute_oviv2_dense_semantics.py \
  --config configs/oviv2_replica_room0_precision_stage2_radseg.json \
  --output /home/ww/oviovo_dense_cache/room0_radseg_b_2f \
  --num-frames 2 --backend radseg \
  --source-root /home/ww/oviovo_references/modules/RADSeg \
  --radio-root /home/ww/oviovo_references/modules/RADIO \
  --model-version c-radio_v3-b --lang-model siglip2 \
  --sample-stride 4 --top-k 4
```

- [ ] **Step 2: Smoke RADSeg-large+SAM on GPU 1**

Use the same command with output `room0_radseg_l_sam_2f`, `CUDA_VISIBLE_DEVICES=1`, model `c-radio_v3-l`, `--sam-refinement`, and `--sam-checkpoint /home/ww/oviovo_benchmark_assets/weights/segment_anything/sam_vit_h_4b8939.pth`.

- [ ] **Step 3: Smoke NARADIO fallback on GPU 2**

Use output `room0_naradio_l_2f`, backend `naradio`, RayFronts source root, model `radio_v2.5-l`, language model `siglip`, and `CUDA_VISIBLE_DEVICES=2`.

- [ ] **Step 4: Verify deterministic cache replay and run mapping**

Regenerate frame 0 for each backend into a separate directory and require identical class IDs plus `allclose(probabilities, atol=1e-4, rtol=1e-4)`. Then point a temporary Stage 2 config at each 2-frame cache and run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/run_oviv2_replica.py \
  --config /tmp/oviv2_stage2_2f.json \
  --output outputs/oviv2_room0_precision_stage2_2f \
  --num-frames 2
```

Expected: snapshot schema 3 reloads, both evaluation heads are finite, dense cache/source hashes appear in metadata and run manifest, and no GPU OOM occurs.

### Task 10: Select The Dense Frontend On A 20-Frame Development Run

**Files:**
- Generated outputs only.

- [ ] **Step 1: Precompute 20 frames for three candidates concurrently**

Run RADSeg-base on GPU 0, RADSeg-large+SAM on GPU 1, and NARADIO-large on GPU 2 with separate cache/output directories and identical `sample_stride=4`, `top_k=4`, vocabulary, and source frames.

- [ ] **Step 2: Run the same OVIV2 backend for all candidates**

For each cache, change only `dense_cache_dir`, run 20 frames, and save metrics under a candidate-specific directory. Do not change integration thresholds between candidates.

- [ ] **Step 3: Select by a frozen rule**

Choose the candidate with the highest finite `evaluation_dense/metrics.json:miou`. Break ties within `0.002` by lower mapping plus inference seconds/frame, then lower peak GPU memory. Reject any candidate whose F@5cm differs from Stage 1 by more than `1e-6` or whose class-agnostic AP50 differs by more than `1e-6`; those metrics must be invariant because only semantics changed.

- [ ] **Step 4: Record the selection artifact**

Write `outputs/oviv2_room0_precision_stage2_selection.json` with candidate config/hash, mIoU/mAcc/f-mIoU, AP25/AP50/F5, inference time, peak GPU memory, selected candidate, and rule version. This is a development artifact and is not a paper result.

### Task 11: Run The 200-Frame Stage 2 Acceptance Gate

**Files:**
- Generated outputs only.

- [ ] **Step 1: Freeze the selected config and precompute 200 frames**

Use exactly the winning 20-frame backend/model/inference/integration settings. Store the cache under an immutable candidate-specific directory and verify all 200 checksums before mapping.

- [ ] **Step 2: Run the 200-frame map**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/run_oviv2_replica.py \
  --config configs/oviv2_replica_room0_precision_stage2_radseg.json \
  --output outputs/oviv2_room0_precision_stage2_200f
```

- [ ] **Step 3: Apply the approved Stage 2 gate**

Require:

```text
dense mIoU > 0.21290644009436308
owner mIoU == 0.21290644009436308 within 1e-9
AP50 >= 0.04977236594883654
F@5cm >= 0.90
all metrics finite
snapshot schema == 3
snapshot and dense cache reload successfully
```

The primary progression gate is dense mIoU `> 0.271`. If the dense head improves Stage 1 but does not exceed `0.271`, retain Stage 2 as an ablation and diagnose prompt/integration errors before Stage 3; do not tune on Replica-7.

### Task 12: Final Regression And Handoff To Stage 3

**Files:**
- No production edits expected.

- [ ] **Step 1: Run whitespace and complete test suites**

```bash
git diff --check 83b04f3..HEAD
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/oviv2 \
  tests/evaluation/test_oviv2_replica.py \
  tests/evaluation/test_evaluate_oviv2_replica_cli.py \
  tests/evaluation/test_run_oviv2_replica_cli.py \
  tests/evaluation/test_precompute_oviv2_dense_semantics.py
```

- [ ] **Step 2: Verify external references remain clean**

```bash
git -C /home/ww/oviovo_references/modules/RADSeg status --short
git -C /home/ww/oviovo_references/modules/RayFronts status --short
git -C /home/ww/oviovo_references/modules/RADIO status --short
```

Expected: no output.

- [ ] **Step 3: Print exact Stage 1 versus Stage 2 metrics**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python - <<'PY'
import json
paths = {
    "stage1": "outputs/oviv2_room0_precision_stage1_200f/evaluation/metrics.json",
    "stage2_owner": "outputs/oviv2_room0_precision_stage2_200f/evaluation_owner/metrics.json",
    "stage2_dense": "outputs/oviv2_room0_precision_stage2_200f/evaluation_dense/metrics.json",
}
for name, path in paths.items():
    data = json.load(open(path))
    print(name, {key: data[key] for key in ("miou", "macc", "f_miou", "ap25", "ap50", "f5")})
PY
```

- [ ] **Step 4: Preserve the benchmark boundary**

Do not update Table 1 from the room0 development run. If Stage 2 passes, write the separate Stage 3 cross-layer fusion plan; only the frozen Replica-8/Replica-7 run may update benchmark result JSON and table tokens.
