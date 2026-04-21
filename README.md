# OVIOVO — Online Open-Vocabulary Instance-First Dynamic Mapping

A research codebase backbone for an online open-vocabulary dynamic mapping system with:
- A global background map (TSDF-like)
- Object-centric local maps with per-instance geometry and semantic memory
- Instance-first, spatial-consistency-first association
- Lifecycle-aware dynamic maintenance

This is the **architectural skeleton only** — placeholder implementations are provided for all modules. Real algorithms are to be filled in iteratively.

## Quick Start

```bash
# Run the end-to-end demo on synthetic data
python run_demo.py

# Run unit tests
python -m pytest tests/ -v
```

Proposal backend switching for comparison:

```bash
# SAM2 proposal overlays
conda run -n ovo python run_replica20f_sam2_overlay.py

# SAM2 vs E-SAM vs EntitySAM comparison
conda run -n ovo python run_replica20f_compare_proposals.py
```

### EntitySAM Backend

[EntitySAM](https://github.com/ymq2017/entitysam) (CVPR'25) extends SAM2 for video entity segmentation. The adapter uses EntitySAM's `SAM2AutomaticMaskGenerator` for per-frame proposal generation with the same interface as vanilla SAM2.

**Setup:**
1. Clone: `git clone https://github.com/ymq2017/entitysam /home/phl/vv/paper2/entitysam`
2. Install: `cd /home/phl/vv/paper2/entitysam && pip install -e .`
3. Download checkpoint: [vit-l](https://huggingface.co/mqye/entitysam/blob/main/checkpoints/vit-l/model_0009999.pth) to `checkpoints/vit-l/model_0009999.pth`

**Config fields** (`configs/default.yaml` under `proposal.entitysam`):
- `repo_root` — path to local EntitySAM repo
- `config_path` — Hydra model config (relative to repo_root)
- `checkpoint_path` — trained weights (relative to repo_root)
- `device`, `points_per_side`, `nms_iou_th`, `stability_score_th`, etc.

**Run comparison:**
```bash
conda run -n ovo python run_replica20f_compare_proposals.py \
    --entitysam-device cuda \
    --entitysam-repo-root /home/phl/vv/paper2/entitysam
# Outputs: outputs/replica20f_compare/{sam2,esam,entitysam}/
```

## Project Structure

```
oviovo/
├── configs/default.yaml              # Central YAML configuration
├── src/
│   ├── core/data_structures.py       # All dataclasses and enums
│   ├── modules/                      # 11 pipeline modules
│   │   ├── frame_input.py            # M1: RGB-D frame packaging
│   │   ├── proposal.py               # M2: Class-agnostic 2D proposals
│   │   ├── runtime_vis.py            # M3: Object-level mask consolidation
│   │   ├── depth_refinement.py       # M4: Depth-based mask refinement
│   │   ├── patch_lifting.py          # M5: 2D→3D patch lifting
│   │   ├── bg_obj_split.py           # M6: Background/object classification
│   │   ├── association.py            # M7: Spatial-first object association
│   │   ├── object_update.py          # M8: Object geometry & history update
│   │   ├── semantic_memory.py        # M9: Object-level semantic memory
│   │   ├── background_update.py      # M10: Background map integration
│   │   └── dynamic_maintenance.py    # M11: Lifecycle & cleanup
│   ├── models/                       # Swappable backends
│   │   ├── proposal_backend.py       # ABC + placeholder for SAM/FastSAM
│   │   ├── sam2_proposal_backend.py  # SAM2 adapter
│   │   ├── esam_proposal_backend.py  # E-SAM adapter
│   │   ├── entitysam_proposal_backend.py  # EntitySAM adapter (CVPR'25)
│   │   └── semantic_backend.py       # ABC + placeholder for SigLIP/CLIP
│   ├── utils/
│   │   ├── geometry.py               # Depth projection, bbox IoU, downsampling
│   │   └── logging.py               # Logging setup
│   └── pipelines/main_pipeline.py    # Central pipeline runner
├── tests/test_pipeline.py            # Unit & integration tests
├── run_demo.py                       # Synthetic data demo
└── README.md
```

## Module Overview

| # | Module | Input | Output | Key Extension Points |
|---|--------|-------|--------|---------------------|
| 1 | Frame Input | RGB, depth, pose | `Frame` | Dataset loaders, ROS integration |
| 2 | Proposal | `Frame.rgb`, `Frame.depth` | `List[Proposal2D]` | SAM, FastSAM backends |
| 3 | Runtime Vis | raw proposals, frame, state | merged `List[Proposal2D]` + explainable grouping metadata | Fragment consolidation, whole-object priors |
| 4 | Depth Refinement | depth, proposals | refined `List[Proposal2D]` | Learned edge detectors |
| 5 | Patch Lifting | proposals, depth, pose, intrinsics | `List[Patch3D]` | Normal estimation |
| 6 | BG/Object Split | patches, background, objects | bg/obj/ambiguous patches | Motion, support-plane analysis |
| 7 | Association | object patches, objects | `AssociationResult` | Learned matching, spatial voting |
| 8 | Object Update | association, patches, state | updated `SystemState` | Local TSDF, ICP refinement |
| 9 | Semantic Memory | state, RGB, updated IDs | updated semantic memories | SigLIP/CLIP, medoid fusion |
| 10 | Background Update | bg patches, background, objects | updated `BackgroundMap` | Full TSDF fusion |
| 11 | Dynamic Maintenance | system state | cleaned state | Bayesian stability, re-identification |

## Core Data Structures

- **`CameraIntrinsics`** — fx, fy, cx, cy, width, height
- **`Frame`** — RGB-D frame with pose and intrinsics
- **`Proposal2D`** — Class-agnostic 2D mask with bbox
- **`Patch3D`** — 3D point cloud patch with centroid and bbox
- **`ObjectState`** — Enum: ACTIVE, INACTIVE, GHOST, REMOVED, DORMANT
- **`SemanticMemory`** — Feature bank, aggregated feature, label hypotheses
- **`ObservationRecord`** — Per-frame observation of an object
- **`ObjectMap`** — Full object instance with geometry, semantics, lifecycle
- **`BackgroundMap`** — Global background (point cloud / TSDF placeholder)
- **`AssociationScore`** — Per-pair scoring breakdown
- **`AssociationResult`** — Matched pairs + new object requests
- **`SystemState`** — Top-level container (objects + background + counters)

## Design Principles

1. **Instance-first**: Build class-agnostic instances first, then layer on semantics
2. **Spatial-first association**: Geometry dominates; semantics are weak auxiliary signals
3. **Dual-layer maps**: Background and objects are maintained separately
4. **Swappable backends**: Proposal and semantic models behind ABCs
5. **Lifecycle-aware**: Objects transition through ACTIVE → INACTIVE → GHOST → REMOVED
6. **Extensible**: Every module has TODO markers for real algorithm insertion

## Runtime Vis

`runtime_vis` sits between raw proposal generation and depth refinement.
Its job is to consolidate over-segmented SAM2 fragments into object-level
2D candidates while keeping the explanation trail:

- `raw_mask_id -> group_id`
- `group_id -> linked_object_id` when a historical object prior fits
- per-edge merge score breakdown and rejection reasons

The core idea is that raw SAM2 masks are high-recall surface fragments,
while downstream lifting and association should operate on more coherent
object candidates. Historical whole-object observations act as a strong
prior boost, but not a hard merge constraint.

## Proposal Backend Comparison

The proposal stage remains backend-swappable behind `ProposalBackend`.
Current supported backend keys are:

- `sam2`
- `esam`

Both are normalized into the same `Proposal2D` output type and carry backend
provenance through:

- `Proposal2D.backend_name`
- `Proposal2D.metadata`

### Config layout

Use `configs/default.yaml`:

- `proposal.backend`
- `proposal.sam2.*`
- `proposal.esam.*`

### E-SAM status on this machine

The wrapper is implemented, but the current `/home/phl/vv/E-SAM` checkout
looks like a project-page repository and does not expose an obvious Python
inference entrypoint. So the current behavior is:

- if you provide a valid local E-SAM adapter entrypoint, it runs
- otherwise the comparison runner writes a clear backend-specific error manifest
  instead of pretending the backend worked

### Minimal E-SAM wiring

Set `proposal.backend: esam`, then provide at least one of:

- `proposal.esam.factory`: `module:function` that returns a callable predictor
- `proposal.esam.predictor`: `module:function` callable predictor symbol

The predictor output should be any of:

- `{"annotations": [{"segmentation": mask, ...}, ...]}`
- `{"masks": (N,H,W), "scores": ..., "boxes": ...}`
- list of dict proposals with `segmentation` or `mask`

All outputs are normalized into `Proposal2D` with `backend_name="esam"`.

## Future Extension Roadmap

- [ ] SAM / FastSAM proposal backend
- [ ] SigLIP / CLIP semantic backend with multi-view fusion
- [ ] Depth-based mask refinement with learned edges
- [ ] Local TSDF per object (replace point cloud accumulation)
- [ ] Full TSDF background map
- [ ] Spatial voting / learned association scoring
- [ ] Bayesian stability checks for lifecycle transitions
- [ ] Re-identification of dormant objects
- [ ] Boundary contamination repair
- [ ] Background reclaim from removed objects

---

## Interface Document

### Per-Module Specifications

#### M1: Frame Input
- **Responsibility**: Package raw inputs into Frame
- **Input types**: `np.ndarray` (rgb, depth), `np.ndarray` (pose), `CameraIntrinsics`
- **Output types**: `Frame`
- **Internal state**: frame counter
- **Extension points**: dataset-specific loaders, ROS subscriber
- **Dependencies**: none

#### M2: Proposal
- **Responsibility**: Class-agnostic 2D segmentation
- **Input types**: `np.ndarray` (rgb), `np.ndarray` (depth)
- **Output types**: `List[Proposal2D]`
- **Internal state**: proposal backend instance
- **Extension points**: `ProposalBackend` ABC — add SAM, FastSAM, etc.
- **Dependencies**: M1 (Frame)

#### M3: Runtime Vis
- **Responsibility**: Merge raw fragment masks into object-level candidates
- **Input types**: `Frame`, `List[Proposal2D]`, `SystemState`
- **Output types**: `RuntimeVisOutput`
- **Internal state**: runtime thresholds
- **Extension points**: pairwise grouping score, whole-object evidence, graph grouping
- **Dependencies**: M1, M2, current state

#### M4: Depth Refinement
- **Responsibility**: Refine masks using depth edges
- **Input types**: `np.ndarray` (depth), `List[Proposal2D]`
- **Output types**: `List[Proposal2D]`
- **Internal state**: edge threshold parameters
- **Extension points**: learned depth edge detector, bilateral filtering
- **Dependencies**: M1, M2, M3

#### M5: Patch Lifting
- **Responsibility**: 2D→3D back-projection
- **Input types**: `List[Proposal2D]`, `np.ndarray` (depth, pose), `CameraIntrinsics`
- **Output types**: `List[Patch3D]`
- **Internal state**: none
- **Extension points**: normal estimation, point filtering
- **Dependencies**: M1, M4

#### M6: BG/Object Split
- **Responsibility**: Classify patches as background/object/ambiguous
- **Input types**: `List[Patch3D]`, `BackgroundMap`, `Dict[int, ObjectMap]`
- **Output types**: three `List[Patch3D]`
- **Internal state**: classification thresholds
- **Extension points**: motion detection, support-plane analysis, semantic hints
- **Dependencies**: M5, current state

#### M7: Association
- **Responsibility**: Match object patches to known instances
- **Input types**: `List[Patch3D]`, `Dict[int, ObjectMap]`
- **Output types**: `AssociationResult`
- **Internal state**: scoring weights
- **Extension points**: geometry overlap computation, spatial voting, learned matching
- **Dependencies**: M6, current state

#### M8: Object Update
- **Responsibility**: Merge patches into objects, create new objects
- **Input types**: `AssociationResult`, `List[Patch3D]`, `SystemState`
- **Output types**: `SystemState`
- **Internal state**: downsample counters
- **Extension points**: local TSDF fusion, ICP pose refinement
- **Dependencies**: M7

#### M9: Semantic Memory
- **Responsibility**: Update per-object semantic features
- **Input types**: `SystemState`, `np.ndarray` (rgb), `List[int]` (updated IDs)
- **Output types**: `SystemState`
- **Internal state**: semantic backend instance
- **Extension points**: `SemanticBackend` ABC — add SigLIP, CLIP; multi-view fusion
- **Dependencies**: M8

#### M10: Background Update
- **Responsibility**: Integrate background geometry
- **Input types**: `List[Patch3D]`, `BackgroundMap`, `Dict[int, ObjectMap]`
- **Output types**: `BackgroundMap`
- **Internal state**: accumulated point cloud / volume
- **Extension points**: TSDF fusion, object occupancy masking
- **Dependencies**: M6

#### M11: Dynamic Maintenance
- **Responsibility**: Lifecycle transitions, ghost cleanup, re-identification
- **Input types**: `SystemState`
- **Output types**: `SystemState`
- **Internal state**: lifecycle parameters
- **Extension points**: Bayesian stability, boundary repair, background reclaim, re-id
- **Dependencies**: all state

### System-Level Pipeline Table

| Step | Module | Input | Output | Notes |
|------|--------|-------|--------|-------|
| 1 | Frame Input | raw RGB-D + pose | `Frame` | Entry point |
| 2 | Proposal | `Frame.rgb`, `Frame.depth` | `List[Proposal2D]` | Swappable backend |
| 3 | Runtime Vis | raw proposals, state | merged proposals + grouping metadata | Fragment consolidation |
| 4 | Depth Refinement | depth, proposals | refined proposals | Geometric filtering |
| 5 | Patch Lifting | proposals, depth, pose | `List[Patch3D]` | Back-projection |
| 6 | BG/Object Split | patches, state | bg/obj/amb lists | Heuristic classification |
| 7 | Association | obj patches, objects | `AssociationResult` | Spatial-first scoring |
| 8 | Object Update | association, patches | updated state | Geometry merge + new creation |
| 9 | Semantic Memory | state, rgb, IDs | updated semantics | Post-association only |
| 10 | Background Update | bg patches, state | updated background | Object-masked integration |
| 11 | Dynamic Maintenance | full state | cleaned state | Periodic lifecycle check |
