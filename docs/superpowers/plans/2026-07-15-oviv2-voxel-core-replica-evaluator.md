# OVIV2 Voxel Core And Replica Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task by task.

**Goal:** Build the voxel-first OVIV2 map substrate and an OVI-MAP-aligned Replica evaluator that can be verified entirely with synthetic fixtures before the real room0 frontend is connected.

**Architecture:** Open3D `VoxelBlockGrid` owns sparse TSDF geometry and Marching Cubes extraction. OVIV2 owns separate sparse block arrays for semantic evidence, competing entity evidence, and reversible ownership in the same 5 cm integer voxel address space. Evaluation projects labeled mesh vertices to Replica GT vertices with a strict 5 cm gate and computes semantic mIoU/mAcc/f-mIoU, semantic instance AP25/AP50, and geometry F@5cm.

**Tech Stack:** Python 3.10, NumPy, SciPy `cKDTree`, Open3D 0.19, pytest. Commands run in the existing `ww-ai` container.

**Execution root:** `/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates`

**Verification prefix:**

```bash
docker exec ww-ai bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && <command>'
```

---

## Task 1: Sparse TSDF Geometry Volume

**Files:**
- Create: `src/oviv2/__init__.py`
- Create: `src/oviv2/geometry.py`
- Create: `tests/oviv2/__init__.py`
- Create: `tests/oviv2/test_geometry.py`

**Contract:**

`SparseTsdfVolume` is the only geometry authority. It accepts metric depth, RGB, intrinsics, and camera-to-world pose; converts Open3D's extrinsic convention internally; integrates float depth and float RGB; extracts a tensor triangle mesh; and saves/loads the native voxel grid. It exposes no point-cloud cache.

Required public API:

```python
@dataclass(frozen=True)
class TsdfConfig:
    voxel_size_m: float = 0.05
    block_resolution: int = 8
    block_count: int = 100_000
    depth_max_m: float = 10.0
    trunc_voxel_multiplier: float = 4.0

class SparseTsdfVolume:
    def __init__(self, config: TsdfConfig = TsdfConfig()) -> None: ...
    def integrate(self, depth_m, rgb, intrinsics, camera_to_world) -> int: ...
    def extract_mesh(self, weight_threshold: float = 1.0) -> open3d.t.geometry.TriangleMesh: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path, config: TsdfConfig) -> SparseTsdfVolume: ...
    @property
    def active_block_count(self) -> int: ...
```

**Step 1: Write failing tests**

Cover:
- invalid voxel size, block resolution, depth limit, and truncation configuration;
- a 48x64 synthetic plane at 1 m allocates blocks and extracts a non-empty mesh;
- invalid depth pixels (`0`, negative, `NaN`, `inf`) do not crash integration;
- camera-to-world is inverted before Open3D integration;
- save/load preserves active block count and extracted mesh vertex count;
- the class has no `points`, `point_cloud`, `dense_map`, or `local_pcd` state.

Run:

```bash
python -m pytest tests/oviv2/test_geometry.py -q
```

Expected: FAIL because `src.oviv2.geometry` does not exist.

**Step 2: Implement the minimum wrapper**

- Validate shapes and finite matrices before calling Open3D.
- Replace invalid depth with zero in a private copy.
- Convert `uint8` RGB to contiguous `float32` in `[0, 1]` so it matches float depth.
- Use `world_to_camera = np.linalg.inv(camera_to_world)` as Open3D extrinsic.
- Use attributes `tsdf`, `weight`, `color`, all `float32`.
- Return the number of unique touched block coordinates from `integrate`.
- Write through a temporary path then `os.replace` in `save`.
- Reconstruct the wrapper around `VoxelBlockGrid.load` in `load` and validate voxel size/block resolution against `TsdfConfig`.

**Step 3: Run focused tests**

```bash
python -m pytest tests/oviv2/test_geometry.py -q
```

Expected: PASS.

**Step 4: Commit only Task 1 files**

```bash
git add src/oviv2/__init__.py src/oviv2/geometry.py tests/oviv2/__init__.py tests/oviv2/test_geometry.py
git commit -m "feat: add OVIV2 sparse TSDF geometry"
```

---

## Task 2: Shared Voxel Addressing And Sparse Evidence Blocks

**Files:**
- Create: `src/oviv2/addressing.py`
- Create: `src/oviv2/evidence.py`
- Create: `tests/oviv2/test_addressing.py`
- Create: `tests/oviv2/test_evidence.py`
- Modify: `src/oviv2/__init__.py`

**Contract:**

Evidence uses global integer voxel keys and sparse 8x8x8 NumPy blocks. It never stores geometry. Semantic and entity candidates retain bounded top-k support instead of collapsing to a single label/owner.

Required public API:

```python
VoxelKey = tuple[int, int, int]
BlockKey = tuple[int, int, int]

def point_to_voxel(point_xyz, voxel_size_m: float) -> VoxelKey: ...
def split_voxel_key(key: VoxelKey, block_resolution: int) -> tuple[BlockKey, tuple[int, int, int]]: ...
def join_voxel_key(block_key, local_key, block_resolution: int) -> VoxelKey: ...

@dataclass(frozen=True)
class EvidenceConfig:
    block_resolution: int = 8
    semantic_top_k: int = 4
    entity_top_k: int = 4

class SparseEvidenceStore:
    def update_semantic(self, voxel_key, label_id: int, support_delta: float, revision: int) -> None: ...
    def update_entity(self, voxel_key, entity_id: int, positive_delta: float, negative_delta: float, timestamp: float, revision: int) -> None: ...
    def semantic_candidates(self, voxel_key) -> tuple[SemanticCandidate, ...]: ...
    def entity_candidates(self, voxel_key) -> tuple[EntityCandidate, ...]: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path, config: EvidenceConfig) -> SparseEvidenceStore: ...
```

**Step 1: Write failing tests**

Cover:
- floor division and modulo round-trip for positive and negative voxel keys;
- point-to-voxel boundary behavior;
- blocks allocate only on first update;
- repeated candidate updates accumulate support;
- top-k eviction is deterministic by support then integer ID;
- positive and negative entity support coexist;
- querying an absent block returns an empty tuple without allocation;
- save/load produces identical candidates and revisions.

Run:

```bash
python -m pytest tests/oviv2/test_addressing.py tests/oviv2/test_evidence.py -q
```

Expected: FAIL because the modules do not exist.

**Step 2: Implement sparse fixed-shape blocks**

- Represent each allocated block with fixed arrays for IDs, supports, timestamps, and revisions.
- Reserve semantic/entity ID `0` for empty; reject non-positive IDs.
- Keep allocation and serialization deterministic by sorting block keys.
- Store schema version, config, block keys, and arrays in compressed NPZ.
- Use temporary-file plus `os.replace` persistence.
- Do not import `src.modules.tsdf_instance_map` or any legacy map class.

**Step 3: Run focused tests**

```bash
python -m pytest tests/oviv2/test_addressing.py tests/oviv2/test_evidence.py -q
```

Expected: PASS.

**Step 4: Commit only Task 2 files**

```bash
git add src/oviv2/__init__.py src/oviv2/addressing.py src/oviv2/evidence.py tests/oviv2/test_addressing.py tests/oviv2/test_evidence.py
git commit -m "feat: add OVIV2 sparse voxel evidence"
```

---

## Task 3: Reversible Ownership And Atomic Voxel Snapshot

**Files:**
- Create: `src/oviv2/ownership.py`
- Create: `src/oviv2/snapshot.py`
- Create: `tests/oviv2/test_ownership.py`
- Create: `tests/oviv2/test_snapshot.py`
- Modify: `src/oviv2/__init__.py`

**Contract:**

Ownership is a derived sparse layer. Releasing or replacing an owner changes ownership only; it must not mutate TSDF or competing evidence. A snapshot directory atomically binds geometry, evidence, ownership, metadata, and checksums.

Required public API:

```python
@dataclass(frozen=True)
class OwnershipRecord:
    entity_id: int
    confidence: float
    epoch: int
    evidence_revision: int

class ReversibleOwnershipStore:
    def assign(self, voxel_key, entity_id: int, confidence: float, evidence_revision: int) -> None: ...
    def release(self, voxel_key, expected_entity_id: int, evidence_revision: int) -> bool: ...
    def owner_of(self, voxel_key) -> OwnershipRecord | None: ...
    def voxels_for_entity(self, entity_id: int) -> frozenset[VoxelKey]: ...

@dataclass(frozen=True)
class VoxelSnapshotMetadata:
    scene_id: str
    frame_id: int
    timestamp: float
    revision: int
    voxel_size_m: float
    block_resolution: int
    schema_version: int = 1

class VoxelMapSnapshot:
    @classmethod
    def commit(cls, target_dir, metadata, geometry, evidence, ownership) -> VoxelMapSnapshot: ...
    @classmethod
    def load(cls, snapshot_dir) -> VoxelMapSnapshot: ...
```

**Step 1: Write failing tests**

Cover:
- assignment, replacement, release, stale expected-owner rejection, epoch increment;
- inverted entity-to-voxel index remains consistent;
- owner release leaves evidence candidates and geometry mesh unchanged;
- snapshot commit creates `metadata.json`, `geometry.npz`, `evidence.npz`, `ownership.npz`, `checksums.json`;
- load rejects a changed checksum, incompatible schema, and mismatched voxel config;
- committing over an existing snapshot is atomic and leaves no temporary directory.

Run:

```bash
python -m pytest tests/oviv2/test_ownership.py tests/oviv2/test_snapshot.py -q
```

Expected: FAIL because ownership/snapshot modules do not exist.

**Step 2: Implement ownership and snapshot**

- Use the shared block addressing but serialize ownership separately from evidence.
- Reject entity ID `0`, non-finite confidence, confidence outside `[0, 1]`, and revision rollback.
- Snapshot commit writes a sibling temporary directory, fsyncs files, computes SHA-256, then swaps the completed directory into place.
- Snapshot load verifies checksums before constructing stores.
- Metadata is immutable and JSON is emitted with sorted keys.

**Step 3: Run focused tests**

```bash
python -m pytest tests/oviv2/test_ownership.py tests/oviv2/test_snapshot.py -q
```

Expected: PASS.

**Step 4: Commit only Task 3 files**

```bash
git add src/oviv2/__init__.py src/oviv2/ownership.py src/oviv2/snapshot.py tests/oviv2/test_ownership.py tests/oviv2/test_snapshot.py
git commit -m "feat: add reversible OVIV2 voxel snapshots"
```

---

## Task 4: Labeled Mesh Derivation

**Files:**
- Create: `src/oviv2/meshing.py`
- Create: `tests/oviv2/test_meshing.py`
- Modify: `src/oviv2/__init__.py`

**Contract:**

Mesh vertices are derived from TSDF and labeled by querying the colocated evidence/ownership layers. The mesh is an evaluation artifact and cannot mutate the voxel map.

Required public API:

```python
@dataclass(frozen=True)
class LabeledMesh:
    vertices_xyz: np.ndarray
    triangles: np.ndarray
    colors_rgb: np.ndarray
    semantic_ids: np.ndarray
    entity_ids: np.ndarray
    semantic_confidence: np.ndarray
    ownership_confidence: np.ndarray

def derive_labeled_mesh(geometry, evidence, ownership, *, weight_threshold: float = 1.0) -> LabeledMesh: ...
def write_labeled_mesh(path: str | Path, mesh: LabeledMesh) -> None: ...
```

**Step 1: Write failing tests**

Cover:
- synthetic plane yields shape-consistent vertex, triangle, color, semantic, and entity arrays;
- mesh vertices take the strongest semantic candidate and current owner from their voxel key;
- unlabeled vertices receive ID `0` and confidence `0`;
- deriving or writing a mesh does not change geometry/evidence/ownership revisions;
- output PLY can be read by Open3D and preserves vertex count.

Run:

```bash
python -m pytest tests/oviv2/test_meshing.py -q
```

Expected: FAIL because `src.oviv2.meshing` does not exist.

**Step 2: Implement the derivation**

- Convert each mesh vertex to the shared global voxel key.
- Select semantics deterministically by support then label ID.
- Read ownership directly; do not infer owner from a semantic label.
- Make all returned NumPy arrays C-contiguous and read-only.
- Write entity ID and semantic ID as explicit PLY vertex properties using Open3D tensor geometry.

**Step 3: Run focused tests**

```bash
python -m pytest tests/oviv2/test_meshing.py -q
```

Expected: PASS.

**Step 4: Commit only Task 4 files**

```bash
git add src/oviv2/__init__.py src/oviv2/meshing.py tests/oviv2/test_meshing.py
git commit -m "feat: derive labeled meshes from OVIV2 voxels"
```

---

## Task 5: OVI-MAP-Aligned Replica Metrics

**Files:**
- Create: `src/evaluation/oviv2_replica.py`
- Create: `tests/evaluation/test_oviv2_replica.py`
- Modify: `src/evaluation/__init__.py`

**Contract:**

Evaluation operates on one `LabeledMesh` plus GT mesh-domain arrays. It uses nearest predicted mesh vertex per GT vertex with strict distance `< 0.05 m`. Instance AP is semantic-class constrained, ignores structural classes, excludes GT/predicted regions smaller than 100 GT-domain vertices, and requires predicted entities to have at least two accepted views.

Required public API:

```python
@dataclass(frozen=True)
class ReplicaGroundTruth:
    vertices_xyz: np.ndarray
    semantic_ids: np.ndarray
    instance_ids: np.ndarray

@dataclass(frozen=True)
class EntityEvaluationInfo:
    entity_id: int
    semantic_id: int
    accepted_view_count: int

def project_mesh_to_gt(mesh: LabeledMesh, gt_vertices_xyz, max_distance_m: float = 0.05) -> ProjectedLabels: ...
def evaluate_replica_voxel_map(mesh, ground_truth, entity_info, *, valid_semantic_ids, instance_semantic_ids, min_instance_vertices=100, distance_threshold_m=0.05) -> dict: ...
```

**Step 1: Write failing tests**

Cover:
- a vertex at `0.049999 m` transfers while one at exactly `0.05 m` does not;
- unmatched GT vertices use semantic/entity ID `0`;
- hand-computed confusion produces exact per-class IoU, accuracy, mIoU, mAcc, and f-mIoU;
- two semantic classes with duplicate entity IDs cannot cross-match;
- predictions with one accepted view or fewer than 100 mapped vertices are excluded;
- AP25 and AP50 match hand-computed fixtures and confidence is area-derived within class;
- geometry precision/recall/F5 uses raw predicted and GT vertices before projection;
- empty prediction returns finite zeros rather than NaN.

Run:

```bash
python -m pytest tests/evaluation/test_oviv2_replica.py -q
```

Expected: FAIL because `src.evaluation.oviv2_replica` does not exist.

**Step 2: Implement metrics without legacy snapshot conversion**

- Use `scipy.spatial.cKDTree` for both projection and geometry distances.
- Build semantic confusion directly from GT-aligned arrays.
- Keep per-class metrics and evaluated support in the result.
- Implement ScanNet-style greedy AP separately per class at 0.25 and 0.50.
- Macro-average only over classes with valid GT instances.
- Return headline keys `miou`, `macc`, `f_miou`, `ap25`, `ap50`, `f5` plus nested audit data.
- Do not call the dense-point `evaluate_static_snapshot` adapter.

**Step 3: Run focused and regression tests**

```bash
python -m pytest tests/evaluation/test_oviv2_replica.py tests/evaluation/test_baseline_static_metrics.py -q
```

Expected: PASS.

**Step 4: Commit only Task 5 files**

```bash
git add src/evaluation/__init__.py src/evaluation/oviv2_replica.py tests/evaluation/test_oviv2_replica.py
git commit -m "feat: add OVI-MAP-style Replica voxel metrics"
```

---

## Task 6: Snapshot Evaluation CLI And Artifact Contract

**Files:**
- Create: `scripts/evaluation/evaluate_oviv2_replica.py`
- Create: `tests/evaluation/test_evaluate_oviv2_replica_cli.py`
- Create: `tests/architecture/test_oviv2_voxel_first.py`
- Modify: `src/evaluation/oviv2_replica.py`

**Contract:**

One CLI invocation loads an immutable OVIV2 snapshot, derives one labeled mesh, loads Replica GT, and writes all six headline metrics plus aligned audit arrays. Runtime code must not import legacy mutable map classes.

CLI:

```bash
python scripts/evaluation/evaluate_oviv2_replica.py \
  --snapshot <snapshot_dir> \
  --entity-info <entities.jsonl> \
  --gt-mesh <semantic_mesh.ply> \
  --gt-info <info_semantic.json> \
  --manifest configs/evaluation/manifests/replica8.json \
  --scene room0 \
  --output <evaluation_dir>
```

**Step 1: Write failing tests**

Cover:
- synthetic snapshot CLI writes `metrics.json`, `per_class_semantic.json`, `per_class_instance_ap.json`, `gt_aligned_semantic_ids.npy`, `gt_aligned_instance_ids.npy`, and `oviv2_instance_mesh.ply`;
- headline JSON keys are finite and deterministic across two runs;
- scene/vocabulary mismatch and missing GT files fail before writing partial results;
- AST/import scan rejects references from `src/oviv2` to `SystemState`, `ObjectMap`, `local_pcd`, `DenseSurfaceMap`, or `tsdf_instance_map`;
- snapshot files do not contain a dense `Nx3` point cloud payload.

Run:

```bash
python -m pytest tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/architecture/test_oviv2_voxel_first.py -q
```

Expected: FAIL because the CLI does not exist.

**Step 2: Implement the CLI and loaders**

- Parse Replica semantic metadata into GT vertex semantic and instance IDs.
- Load the frozen scene/vocabulary/alias settings from the manifest.
- Write into a temporary evaluation directory and atomically publish it.
- Emit sorted, indented JSON with protocol fields: projection threshold, strict comparator, min region size, vocabulary hash, snapshot revision, and snapshot checksums.
- Preserve the final voxel snapshot when evaluation fails.

**Step 3: Run focused and full new-suite tests**

```bash
python -m pytest tests/oviv2 tests/evaluation/test_oviv2_replica.py tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/architecture/test_oviv2_voxel_first.py -q
```

Expected: PASS.

**Step 4: Run existing architecture/evaluation regressions**

```bash
python -m pytest tests/architecture tests/evaluation -q
```

Expected: PASS, except tests requiring absent external baseline artifacts must be reported explicitly rather than hidden.

**Step 5: Commit only Task 6 files**

```bash
git add scripts/evaluation/evaluate_oviv2_replica.py src/evaluation/oviv2_replica.py tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/architecture/test_oviv2_voxel_first.py
git commit -m "feat: evaluate OVIV2 voxel snapshots on Replica"
```

---

## Completion Gate

This plan is complete when all six task commits exist and the following command passes in `ww-ai`:

```bash
python -m pytest tests/oviv2 tests/evaluation/test_oviv2_replica.py tests/evaluation/test_evaluate_oviv2_replica_cli.py tests/architecture/test_oviv2_voxel_first.py -q
```

The next plan may then connect the existing `ReplicaRoom0Dataset`, frontend adapter, LocalTrack, association, lifecycle, and commit loop to this substrate. It must first pass a two-frame room0 smoke run, then 20 frames, before the frozen 200-frame Table 1 run.
