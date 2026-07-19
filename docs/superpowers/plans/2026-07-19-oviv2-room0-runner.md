# OVIV2 Replica room0 Runner Implementation Plan

> **Execution:** Apply `superpowers:test-driven-development` task by task, then use `superpowers:verification-before-completion` for the room0 acceptance gates.

**Goal:** Build the OVIV2-native runtime that consumes frozen YOLO+SAM frontend detections, processes Replica room0 at 2/20/200 frames, publishes immutable voxel artifacts, and evaluates all six Table 1 metrics.

**Architecture:** The runner converts cached masks and RGB-D frames into typed OVIV2 observations, promotes them through a bounded local tracker, resolves persistent entities with geometry-first voxel overlap, and writes semantic/entity evidence plus reversible ownership beside full-frame TSDF geometry. Legacy proposal refinement and lifting may be called as stateless functions, but no legacy mutable map state crosses the boundary.

**Tech Stack:** Python 3.10, NumPy, Open3D tensor VoxelBlockGrid, SciPy, Pillow, plyfile, pytest.

---

### Task 1: Frozen frontend adapter and typed observations

**Files:**
- Create: `src/oviv2/observations.py`
- Create: `tests/oviv2/test_observations.py`
- Modify: `src/oviv2/__init__.py`

- [ ] Add failing tests for gzip-pickle cache validation, Replica-41 label normalization, explicit `OBJECT`/`STRUCTURE`/`UNKNOWN` kinds, deterministic mask lifting, and empty frames.
- [ ] Implement immutable observation records and a one-way cache adapter that reads only the requested frame.
- [ ] Verify: `pytest -q tests/oviv2/test_observations.py`.

### Task 2: Bounded local tracking and persistent entity registry

**Files:**
- Create: `src/oviv2/tracking.py`
- Create: `src/oviv2/entities.py`
- Create: `tests/oviv2/test_tracking.py`
- Create: `tests/oviv2/test_entities.py`
- Modify: `src/oviv2/__init__.py`

- [ ] Add failing tests for 3-5 frame windows, two-hit promotion, expiry, geometry-first matching, semantic tie breaking, stable entity IDs, and entity JSONL round trips.
- [ ] Implement local tracks without persistent IDs and an entity registry without point-cloud storage.
- [ ] Verify: `pytest -q tests/oviv2/test_tracking.py tests/oviv2/test_entities.py`.

### Task 3: OVIV2 runtime fusion and ownership

**Files:**
- Create: `src/oviv2/runtime.py`
- Create: `tests/oviv2/test_runtime.py`
- Modify: `src/oviv2/__init__.py`

- [ ] Add failing tests proving full-frame geometry integration is independent of observations, structure labels never create entities, tentative objects cannot write entity evidence, and competing evidence survives ownership changes.
- [ ] Implement the fixed runtime order: geometry, observation normalization, tracking, association, evidence fusion, ownership recomputation, lifecycle update, commit metadata.
- [ ] Verify: `pytest -q tests/oviv2/test_runtime.py tests/architecture/test_oviv2_voxel_first.py`.

### Task 4: Atomic room0 runner and provenance artifacts

**Files:**
- Create: `scripts/run_oviv2_replica.py`
- Create: `configs/oviv2_replica_room0.json`
- Create: `tests/evaluation/test_run_oviv2_replica_cli.py`

- [ ] Add failing CLI tests for preflight failures, exact frame selection, atomic snapshots, resume compatibility, entity JSONL, timing, run manifest, hashes, and absence of dense point-cloud state.
- [ ] Implement 2/20/200-frame execution, periodic checkpoints, restore, final mesh derivation, evaluator invocation, and artifact checksums.
- [ ] Verify: `pytest -q tests/evaluation/test_run_oviv2_replica_cli.py`.

### Task 5: Real Replica room0 gates

**Files:**
- Generate: `outputs/oviv2_room0_smoke2/`
- Generate: `outputs/oviv2_room0_smoke20/`
- Generate: `outputs/oviv2_room0_200f/`

- [ ] Run the real two-frame contract smoke and validate restore/evaluation.
- [ ] Run the real twenty-frame system smoke and require multiple accepted entities, a nonempty mesh, and finite metrics.
- [ ] Run all 200 sampled frames at stride 10 and publish the complete output contract.
- [ ] Re-evaluate the final immutable snapshot and compare aligned arrays plus metrics byte-for-byte.

### Task 6: Final verification

**Files:**
- Modify only if a verification failure exposes an OVIV2 defect.

- [ ] Run `pytest -q tests/oviv2 tests/architecture tests/evaluation`.
- [ ] Validate the final manifest hashes and artifact checksums.
- [ ] Confirm the snapshot contains sparse geometry/evidence/ownership plus entities and no dense point cloud.
- [ ] Record the six finite room0 metrics from `evaluation/metrics.json`.
