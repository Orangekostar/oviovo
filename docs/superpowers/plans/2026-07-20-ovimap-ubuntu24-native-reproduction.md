# OVI-MAP Ubuntu 24.04 Native Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the released CropFormer and OVI-MAP pipeline natively on Ubuntu 24.04, prove the mask/color/raycast interfaces are correct, rerun Replica-8, and bind only independently verified metrics to the benchmark table.

**Architecture:** Use separate Conda environments for the official Python 3.8/PyTorch 2.1.1 CropFormer frontend and the Python 3.11 RoboStack Noetic mapping backend. Build fresh hash-bound source copies, apply only recorded Ubuntu 24.04 compatibility patches, and place pure validation logic in the benchmark repository. A staged runner stops at environment, frontend, mapping, room0, Replica-8, and evaluation gates before publishing immutable manifests.

**Tech Stack:** Ubuntu 24.04, Conda, Python 3.8/3.11, PyTorch 2.1.1, CUDA 12.1, Detectron2, CropFormer, RoboStack ROS Noetic, catkin, pybind11, OVI-MAP, NumPy, OpenCV, pytest.

---

## File Structure

- Create `configs/environments/ovimap_cropformer.yaml`: direct frontend dependencies.
- Create `configs/environments/ovimap_map.yaml`: direct RoboStack mapping dependencies.
- Create `scripts/reproduction/ovimap/bootstrap_native_envs.sh`: environments, fresh sources, and provenance.
- Create `patches/cropformer/ubuntu24-torch21.patch`: strict CUDA dispatch and instance-ID exporter.
- Create `patches/ovimap/ubuntu24-native.patch`: build-only changes derived from `paper2`.
- Create `src/evaluation/baselines/ovimap_native.py`: pure interface and gate validation.
- Create `tests/evaluation/test_ovimap_native.py`: mask/color/raycast/bbox contract tests.
- Create `scripts/evaluation/audit_ovimap_native_frame.py`: atomic per-frame audit CLI.
- Create `scripts/evaluation/run_ovimap_native.py`: staged runner and scene manifests.
- Create `tests/evaluation/test_run_ovimap_native.py`: bootstrap, patch, runner, and gate tests.
- Create `scripts/evaluation/finalize_ovimap_native.py`: Replica-8 verification and result candidate.
- Create `tests/evaluation/test_finalize_ovimap_native.py`: result-eligibility tests.
- Modify benchmark report, tokens, and derived tables only after final verification.

Generated builds and runs live outside Git:

```text
/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/
/home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native/
```

The dirty `/home/ww/vv/paper2` checkouts are read-only references.

### Task 1: Encode Native Interface Contracts

**Files:**
- Create: `tests/evaluation/test_ovimap_native.py`
- Create: `src/evaluation/baselines/ovimap_native.py`
- Create: `scripts/evaluation/audit_ovimap_native_frame.py`

- [ ] **Step 1: Write failing mask and bbox tests**

```python
def test_compose_instance_id_mask_uses_score_order():
    masks = np.array([
        [[1, 1], [0, 0]],
        [[0, 1], [0, 1]],
    ], dtype=bool)
    result = compose_instance_id_mask(masks, np.array([0.9, 0.6]))
    assert result.dtype == np.uint8
    assert result.tolist() == [[1, 1], [0, 2]]
    assert sorted(np.unique(result).tolist()) == [0, 1, 2]

def test_bbox_from_instance_map_preserves_sparse_extent():
    ids = np.zeros((4, 5), dtype=np.uint32)
    ids[1:3, 2:4] = 7
    assert bbox_from_instance_map(ids, 7) == [2, 1, 3, 2]
```

Also assert rejection of RGB masks, floats, invalid shapes, non-contiguous positive IDs, wrong raycast buffer length, and RGB/BGR color mismatch.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_native.py -q
```

Expected: FAIL because `src.evaluation.baselines.ovimap_native` does not exist.

- [ ] **Step 3: Implement pure validators**

```python
def _integer_map(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a two-dimensional integer array")
    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative IDs")
    return array


def compose_instance_id_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    scores = np.asarray(scores)
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError("masks must have shape [N,H,W] and boolean dtype")
    if scores.shape != (masks.shape[0],) or not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain one finite value per mask")
    if masks.shape[0] > np.iinfo(np.uint8).max:
        raise ValueError("too many masks for released uint8 instance IDs")
    output = np.zeros(masks.shape[1:], dtype=np.uint8)
    for index in np.argsort(scores, kind="stable"):
        output[masks[index]] = int(index) + 1
    return output


def summarize_instance_id_mask(mask: np.ndarray) -> dict[str, object]:
    mask = _integer_map(mask, name="instance mask")
    ids, counts = np.unique(mask, return_counts=True)
    positive = ids[ids > 0]
    if positive.size and positive.tolist() != list(range(1, int(positive[-1]) + 1)):
        raise ValueError("positive instance IDs must be contiguous")
    foreground = counts[ids > 0]
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "instance_ids": positive.astype(int).tolist(),
        "instance_count": int(positive.size),
        "foreground_ratio": float(foreground.sum() / mask.size),
        "largest_instance_ratio": float(foreground.max() / mask.size) if foreground.size else 0.0,
    }


def normalize_raycast_ids(value: object, *, height: int, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1 and array.size == height * width:
        array = array.reshape(height, width)
    if array.shape != (height, width):
        raise ValueError("raycast ID map shape mismatch")
    return _integer_map(array, name="raycast ID map")


def bbox_from_instance_map(ids: np.ndarray, instance_id: int) -> list[int]:
    ids = _integer_map(ids, name="instance map")
    ys, xs = np.nonzero(ids == instance_id)
    if xs.size == 0:
        raise ValueError(f"instance {instance_id} is absent")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def validate_color_round_trip(
    mapper_rgb: tuple[int, int, int], logged_rgb: tuple[int, int, int]
) -> None:
    for color in (mapper_rgb, logged_rgb):
        if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
            raise ValueError("instance colors must be RGB uint8 triples")
    if tuple(mapper_rgb) != tuple(logged_rgb):
        raise ValueError("mapper and logged instance colors differ")


def audit_native_frame(
    mask: np.ndarray, raycast: np.ndarray, color_pairs: list[dict]
) -> dict[str, object]:
    mask_summary = summarize_instance_id_mask(mask)
    raycast = normalize_raycast_ids(raycast, height=mask.shape[0], width=mask.shape[1])
    for pair in color_pairs:
        validate_color_round_trip(tuple(pair["mapper_rgb"]), tuple(pair["logged_rgb"]))
    boxes = [
        bbox_from_instance_map(raycast, int(instance_id))
        for instance_id in np.unique(raycast)
        if instance_id > 0
    ]
    full = [0, 0, mask.shape[1] - 1, mask.shape[0] - 1]
    full_count = sum(box == full for box in boxes)
    if boxes and full_count == len(boxes):
        raise ValueError("all non-empty raycast boxes are full-frame")
    return {
        "status": "PASS",
        "mask": mask_summary,
        "raycast_instance_count": len(boxes),
        "full_frame_bbox_count": full_count,
        "boxes": boxes,
    }
```

`normalize_raycast_ids` accepts only an integer `H x W` array or an integer buffer of exactly `H * W`. It never takes a color channel or converts RGB to grayscale. `audit_native_frame` returns `PASS` only when IDs, colors, raycasts, and non-empty bboxes are valid.

- [ ] **Step 4: Implement the audit CLI**

Accept `--mask`, `--raycast-npy`, `--colors-json`, `--output`, `--height`, and `--width`; hash inputs, atomically write JSON, and exit nonzero on `FAIL`.

- [ ] **Step 5: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_native.py -q
git add src/evaluation/baselines/ovimap_native.py \
  scripts/evaluation/audit_ovimap_native_frame.py \
  tests/evaluation/test_ovimap_native.py
git commit -m "feat: validate OVI-MAP native interfaces"
```

Expected: PASS and a commit containing only the three listed files.

### Task 2: Bootstrap Two Native Environments

**Files:**
- Create: `configs/environments/ovimap_cropformer.yaml`
- Create: `configs/environments/ovimap_map.yaml`
- Create: `scripts/reproduction/ovimap/bootstrap_native_envs.sh`
- Create: `tests/evaluation/test_run_ovimap_native.py`

- [ ] **Step 1: Write failing bootstrap-contract tests**

Assert the frontend pins Python 3.8.10, PyTorch 2.1.1, torchvision 0.16.1, `pytorch-cuda=12.1`, and `cuda-nvcc=12.1.105`. Assert the mapper pins Python 3.11, NumPy 1.26, `ros-noetic-ros-base=1.5.0`, `ros-noetic-pcl-ros=1.7.4`, catkin tools, CMake, pybind11, protobuf, glog, and gflags.

Assert the script refuses an existing build root and writes `conda-explicit.txt`, `pip-freeze.txt`, `source-hashes.json`, `nvidia-smi.txt`, and `host.json`.

```python
def test_environment_pins_are_exact():
    frontend = yaml.safe_load(Path("configs/environments/ovimap_cropformer.yaml").read_text())
    mapper = yaml.safe_load(Path("configs/environments/ovimap_map.yaml").read_text())
    assert "python=3.8.10" in frontend["dependencies"]
    assert "pytorch=2.1.1" in frontend["dependencies"]
    assert "cuda-nvcc=12.1.105" in frontend["dependencies"]
    assert "python=3.11" in mapper["dependencies"]
    assert "ros-noetic-ros-base=1.5.0" in mapper["dependencies"]
    assert "ros-noetic-pcl-ros=1.7.4" in mapper["dependencies"]

def test_bootstrap_refuses_existing_root(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    completed = subprocess.run(
        ["bash", "scripts/reproduction/ovimap/bootstrap_native_envs.sh", str(root)],
        text=True, capture_output=True,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_native.py -q
```

Expected: FAIL because the environment files and script are absent.

- [ ] **Step 3: Add exact environment definitions**

```yaml
# configs/environments/ovimap_cropformer.yaml
channels: [pytorch, nvidia, conda-forge]
dependencies:
  - python=3.8.10
  - pytorch=2.1.1
  - torchvision=0.16.1
  - pytorch-cuda=12.1
  - cuda-nvcc=12.1.105
  - ninja
  - pip
  - pip:
      - transformers==4.49.0
      - opencv-contrib-python==4.11.*
```

```yaml
# configs/environments/ovimap_map.yaml
channels: [robostack-noetic, conda-forge]
dependencies:
  - python=3.11
  - numpy=1.26
  - ros-noetic-ros-base=1.5.0
  - ros-noetic-pcl-ros=1.7.4
  - catkin_tools
  - cmake
  - make
  - pybind11
  - protobuf
  - glog
  - gflags
  - pip
```

- [ ] **Step 4: Implement and execute bootstrap**

Clone exact source commits into a temporary directory and atomically rename only after verification:

```text
OVI-MAP 58a804e2d7c82ba05a489eb071aba3367301fed8
Entity  6e7e13ac91ef508088e1b848167c01f19b00b512
```

Run:

```bash
bash scripts/reproduction/ovimap/bootstrap_native_envs.sh \
  /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native
```

Expected: both environments and all provenance files exist. Solver failure stops without relaxing pins.

- [ ] **Step 5: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_native.py -q
git add configs/environments/ovimap_cropformer.yaml \
  configs/environments/ovimap_map.yaml \
  scripts/reproduction/ovimap/bootstrap_native_envs.sh \
  tests/evaluation/test_run_ovimap_native.py
git commit -m "build: define native OVI-MAP environments"
```

### Task 3: Build Strict CropFormer CUDA and Exporter

**Files:**
- Create: `patches/cropformer/ubuntu24-torch21.patch`
- Modify: `tests/evaluation/test_run_ovimap_native.py`

- [ ] **Step 1: Write failing patch-content tests**

Assert the patch changes both CUDA dispatch calls from `value.type()` to `value.scalar_type()`, contains no `MSDA = None` or fallback branch, comments out only mmcv-dependent dataset imports, writes single-channel PNG masks, and writes per-frame JSON diagnostics.

```python
def test_cropformer_patch_requires_cuda_without_fallback():
    patch = Path("patches/cropformer/ubuntu24-torch21.patch").read_text()
    assert patch.count("value.scalar_type()") == 2
    assert "MSDA = None" not in patch
    assert "if MSDA is None" not in patch
    assert "demo_cropformer/demo_from_dirs.py" in patch
    assert "frame_diagnostics" in patch
```

- [ ] **Step 2: Create and apply the strict patch**

Apply only to the fresh Entity copy:

```bash
git -C /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/Entity \
  apply --check "$PWD/patches/cropformer/ubuntu24-torch21.patch"
git -C /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/Entity \
  apply "$PWD/patches/cropformer/ubuntu24-torch21.patch"
```

Install Detectron2 `v0.6` at commit `d1e04565d3bec8719335b88be9e9b961bf3ec464`, then build CropFormer's `MultiScaleDeformableAttention` with the new environment's `nvcc`.

- [ ] **Step 3: Prove the CUDA operator executes**

Run forward and backward through `MSDeformAttnFunction` and assert:

```python
assert MSDA is not None
assert MSDA.__file__.endswith('.so')
assert output.is_cuda
assert value.grad is not None
```

Record extension SHA-256 and `ldd`; expected no `not found` entries.

- [ ] **Step 4: Run one official HoRNet frame**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ww/miniconda3/envs/ovimap-cropformer/bin/python \
  /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/Entity/Entityv2/CropFormer/demo_cropformer/demo_from_dirs.py \
  --config-file /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml \
  --input /home/ww/oviovo_baseline_builds/replica_repaired/room0/results/frame000000.jpg \
  --output /home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native/gate2_frontend \
  --confidence-threshold 0.5 --out-type 0 \
  --opts MODEL.WEIGHTS /home/ww/vv/paper2/Entity/checkpoints/CropFormer_hornet_3x_03823a.pth
```

Expected: `680 x 1200` integer mask, background 0, contiguous positive IDs, nonzero foreground, and matching diagnostics.

- [ ] **Step 5: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_native.py -q
git add patches/cropformer/ubuntu24-torch21.patch \
  tests/evaluation/test_run_ovimap_native.py
git commit -m "fix: build strict CropFormer CUDA frontend"
```

### Task 4: Rebuild the OVI-MAP Python Backend

**Files:**
- Create: `patches/ovimap/ubuntu24-native.patch`
- Modify: `tests/evaluation/test_run_ovimap_native.py`

- [ ] **Step 1: Write failing backend-patch tests**

Assert the patch contains only C++17, removal of `gsm_node`/RViz-only dependencies, conda PCL/OpenGL discovery, pybind11/Python 3.11 selection, local SigLIP support, and CLI-controlled dumps of mapper RGB, logged RGB, raycast ID maps, and derived boxes. Reject changes to thresholds, integration, view selection, or evaluator equations.

```python
def test_ovimap_patch_does_not_change_algorithm_constants():
    patch = Path("patches/ovimap/ubuntu24-native.patch").read_text()
    for forbidden in (
        "connection_ratio_th_", "data_association", "inst_association",
        "seg_graph_confidence", "eval_inst_seg.py", "eval_sem_seg.py",
    ):
        assert forbidden not in patch
    assert "CMAKE_CXX_STANDARD" in patch
    assert "native_audit_dir" in patch
    assert "raycast" in patch
```

- [ ] **Step 2: Apply and build**

```bash
git -C /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP \
  apply --check "$PWD/patches/ovimap/ubuntu24-native.patch"
git -C /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP \
  apply "$PWD/patches/ovimap/ubuntu24-native.patch"
cd /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP/mapping_ros_ws
source /home/ww/miniconda3/envs/ovimap-map/setup.bash
catkin config --extend /home/ww/miniconda3/envs/ovimap-map \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
  -DPYTHON_EXECUTABLE=/home/ww/miniconda3/envs/ovimap-map/bin/python \
  -DPython3_EXECUTABLE=/home/ww/miniconda3/envs/ovimap-map/bin/python
catkin config --skiplist voxblox_rviz_plugin gsm_node global_segment_map_node
catkin build consistent_gsm depth_segmentation_py --no-status --summarize
```

- [ ] **Step 3: Verify both Python extensions**

Import `consistent_gsm` and `depth_segmentation_py` with only new Conda/catkin paths. Fail `ldd` if any line contains `not found`, `/home/ww/vv/paper2`, deleted `cird`, or CPython 3.10.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_native.py -q
git add patches/ovimap/ubuntu24-native.patch \
  tests/evaluation/test_run_ovimap_native.py
git commit -m "build: rebuild OVI-MAP backend on Ubuntu 24"
```

### Task 5: Implement the Staged Native Runner

**Files:**
- Create: `scripts/evaluation/run_ovimap_native.py`
- Modify: `tests/evaluation/test_run_ovimap_native.py`

- [ ] **Step 1: Write failing runner tests**

Assert commands generate frames `0..1990` at step 10, keep frontend/mapping Python executables separate, use Replica-51/SigLIP, and contain neither Docker nor a fallback. Assert the only state transitions are:

```text
ENV_PASS -> FRONTEND_PASS -> MAPPING_PASS -> ROOM0_PASS -> REPLICA8_PASS -> EVAL_PASS
```

Any command failure, missing artifact, color mismatch, malformed raycast, or all-full-frame bbox audit stops the transition.

```python
def test_runner_freezes_official_frame_protocol(native_config):
    commands = build_scene_commands(native_config, scene="room0")
    assert commands.frame_ids == list(range(0, 2000, 10))
    combined = " ".join(word for command in commands.commands for word in command)
    assert "--data_association 2" in combined
    assert "--inst_association 4" in combined
    assert "docker" not in combined.lower()

def test_failed_audit_cannot_advance_mapping_gate():
    with pytest.raises(GateFailure, match="native frame audit"):
        advance_state("FRONTEND_PASS", event={"status": "FAIL"})
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_native.py -q
```

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement runner and manifests**

Implement `preflight`, `run_frontend`, `run_mapping`, `audit_scene`, and `write_scene_manifest`. Use subprocess argument lists, separate logs, explicit exit codes, timestamps, SHA-256 records, and new run directories with `exist_ok=False`.

Retain released mapper parameters exactly:

```text
--dataset replica --task Nyu40 --start 0 --end 2000 --step 10
--data_association 2 --inst_association 4 --seg_graph_confidence 3
--save_temp_results --save_temp_geometrics --num_threads 10
```

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_native.py -q
git add scripts/evaluation/run_ovimap_native.py \
  tests/evaluation/test_run_ovimap_native.py
git commit -m "feat: run staged native OVI-MAP reproduction"
```

### Task 6: Pass One-Frame and Room0 Gates

**Files:**
- Generate: `/home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native/gate3_mapping/**`
- Generate: `/home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native/room0/**`

- [ ] **Step 1: Run one-frame mapping**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/run_ovimap_native.py --stage mapping --scene room0 \
  --start 0 --end 1 --step 1 \
  --run-root /home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native
```

Expected: `MAPPING_PASS`; exact color round trip, `680 x 1200` integer raycast map, and at least one non-empty, non-full-frame bbox.

- [ ] **Step 2: Diagnose failures test-first**

Use recorded mask, color pairs, raycast `.npy`, and pybind types to find the first boundary changing data. Add a failing regression test before the fix. Never change mapping parameters.

- [ ] **Step 3: Run room0 200 frames**

```bash
CUDA_VISIBLE_DEVICES=0 /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/run_ovimap_native.py --stage room --scene room0 \
  --start 0 --end 2000 --step 10 \
  --run-root /home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native
```

Expected: `ROOM0_PASS`, 200 masks, final mesh/feature pickle, complete diagnostics, no grayscale color collapse, and full-frame bbox ratio below 1.0.

- [ ] **Step 4: Freeze room0 evidence**

Hash all inputs, logs, masks, mapper artifacts, and audit JSON into `room0/native_mapping_manifest.json`.

### Task 7: Run Replica-8 and Released Evaluation

**Files:**
- Generate: seven remaining scene directories under the run root.
- Existing: `scripts/evaluation/run_ovimap_paper_protocol.py`
- Existing: `scripts/evaluation/audit_ovimap_paper_parity.py`

- [ ] **Step 1: Run seven scenes after room0 passes**

Use at most three concurrent processes, one per A40. Each process uses identical hashes. A failure writes a new attempt directory rather than overwriting output.

```bash
CUDA_VISIBLE_DEVICES=0 /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/evaluation/run_ovimap_native.py --stage room --scene office0 --start 0 --end 2000 --step 10 --run-root /home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native
CUDA_VISIBLE_DEVICES=1 /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/evaluation/run_ovimap_native.py --stage room --scene office1 --start 0 --end 2000 --step 10 --run-root /home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native
CUDA_VISIBLE_DEVICES=2 /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/evaluation/run_ovimap_native.py --stage room --scene office2 --start 0 --end 2000 --step 10 --run-root /home/ww/oviovo_baseline_runs/20260720_ovimap_ubuntu24_native
```

Continue with `office3`, `office4`, `room1`, and `room2`. Expected: eight passing scene manifests.

- [ ] **Step 2: Aggregate native mapping evidence**

Write status `COMPLETE_NATIVE_MAPPING`, exact scene order `office0..office4,room0..room2`, and 200 frame IDs per scene. Reject mixed source, weight, environment, or protocol hashes.

- [ ] **Step 3: Run released Replica-51 evaluation**

Use `scripts/evaluation/run_ovimap_paper_protocol.py` with the new outputs and authoritative `LogInstanceColor`. Expected: all 18 commands pass and semantic mIoU/mAcc plus semantic-instance AP diagnostics are finite.

- [ ] **Step 4: Run paper-parity audit**

Run `scripts/evaluation/audit_ovimap_paper_parity.py` with all eight new feature pickles. Released semantic evidence may pass; Table 2 class-agnostic AP remains unavailable without a separately validated evaluator manifest.

### Task 8: Finalize Results and Regenerate the Table

**Files:**
- Create: `tests/evaluation/test_finalize_ovimap_native.py`
- Create: `scripts/evaluation/finalize_ovimap_native.py`
- Create: `docs/paper/results/baselines/ovimap/replica/20260720-ubuntu24-native/result.json`
- Create: `docs/paper/results/baselines/ovimap/replica/20260720-ubuntu24-native/paper_parity_audit.json`
- Modify: `docs/paper/BASELINE_RUN_REPORT.md`
- Modify: `docs/paper/benchmark_tokens.tsv`
- Modify: `docs/paper/benchmark_tables_baselines.md`
- Modify: `docs/paper/benchmark_tables_baselines.tex`

- [ ] **Step 1: Write failing finalizer tests**

Reject incomplete scenes, mismatched hashes, failed gates, non-finite metrics, renamed precision/recall, and AP25/AP50 binding without a validated `class_agnostic_ap_manifest`. Preserve verified released semantic mIoU/mAcc as diagnostic evidence, but keep all T1 OVI-MAP tokens `UNFILLED` unless the complete paper-parity audit passes. f-mIoU/F5 and unavailable AP always remain `UNFILLED`.

```python
def test_finalizer_keeps_table2_unfilled_without_ap_manifest(complete_fixture):
    complete_fixture["protocol_evidence"].pop("class_agnostic_ap_manifest")
    result = finalize_native_result(complete_fixture)
    assert result["availability"]["class_agnostic_ap25"] == (
        "UNFILLED_NO_RELEASED_EVALUATOR"
    )
    assert result["table_bindings"] == {}

def test_finalizer_rejects_mixed_scene_hashes(complete_fixture):
    complete_fixture["scenes"]["room2"]["source_sha256"] = "0" * 64
    with pytest.raises(FinalizationError, match="source hash mismatch"):
        finalize_native_result(complete_fixture)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_finalize_ovimap_native.py -q
```

Expected: FAIL because the finalizer is absent.

- [ ] **Step 3: Implement and run finalization**

Re-hash external manifests, call `audit_paper_parity`, copy only JSON evidence, and emit explicit availability:

```json
{
  "released_semantic_miou": "VERIFIED_DIAGNOSTIC",
  "released_semantic_macc": "VERIFIED_DIAGNOSTIC",
  "t1_replica8_miou": "UNFILLED_PAPER_AUDIT_INCOMPLETE",
  "t1_replica8_macc": "UNFILLED_PAPER_AUDIT_INCOMPLETE",
  "class_agnostic_ap25": "UNFILLED_NO_RELEASED_EVALUATOR",
  "class_agnostic_ap50": "UNFILLED_NO_RELEASED_EVALUATOR",
  "fmiou": "UNFILLED_NOT_DEFINED_BY_RELEASED_PROTOCOL",
  "f5": "UNFILLED_NOT_DEFINED_BY_RELEASED_PROTOCOL"
}
```

- [ ] **Step 4: Import eligible values and regenerate tables**

Run the existing importer with the previously valid result set. Pass the new OVI-MAP result only if its complete paper-parity audit is `PASS`; otherwise preserve it as diagnostic evidence and confirm every OVI-MAP T1 cell remains a token.

- [ ] **Step 5: Verify and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q \
  tests/evaluation/test_ovimap_native.py \
  tests/evaluation/test_run_ovimap_native.py \
  tests/evaluation/test_finalize_ovimap_native.py \
  tests/evaluation/test_ovimap_paper_audit.py \
  tests/evaluation/test_run_ovimap_paper_protocol.py \
  tests/evaluation/test_import_benchmark_results.py \
  tests/test_benchmark_table_package.py
git diff --check
git add scripts/evaluation/finalize_ovimap_native.py \
  tests/evaluation/test_finalize_ovimap_native.py \
  docs/paper/results/baselines/ovimap/replica/20260720-ubuntu24-native/result.json \
  docs/paper/results/baselines/ovimap/replica/20260720-ubuntu24-native/paper_parity_audit.json \
  docs/paper/BASELINE_RUN_REPORT.md docs/paper/benchmark_tokens.tsv \
  docs/paper/benchmark_tables_baselines.md \
  docs/paper/benchmark_tables_baselines.tex
git commit -m "results: publish verified Ubuntu 24 OVI-MAP rerun"
```

Stage only the listed new implementation, result evidence, report, registry, and derived tables. Do not stage pre-existing dirty pipeline/config files.
