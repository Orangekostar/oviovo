# SED++ Frontend Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in SED++ frontend that replaces the current YOLOWorld + SAM2 + anchor-guided fusion path, with the explicit objective of reducing mapping time while keeping 200-frame stride-10 mIoU/f-mIoU close to the current online baseline.

**Architecture:** Implement SED++ as a `ProposalBackend` named `sedpp`. It should emit class-aware `Proposal2D` masks directly from semantic segmentation components. Disable `anchor_frontend` and `anchor_guided_sam` in the SED++ config so downstream mapping receives proposals without the expensive YOLO/SAM fusion stage. Keep the current YOLOWorld+SAM2 config unchanged for A/B comparison.

**Reference Baseline:** `outputs/tmp_validation/20260612_wwai_online_yoloworld_sam_baseline_s10_200f`, mIoU `0.5190`, f-mIoU `0.6052`, wall `40:37.90`, `proposal_generation` mean `10.4371s`, `anchor_guided_sam_fusion` mean `9.0352s`.

---

## Evidence And Constraints

- Local SED implementation exists at `/home/ww/vv/paper2/SED`.
- Local SED weights exist:
  - `/home/ww/vv/paper2/SED/weights/sed_model_large.pth`
  - `/home/ww/vv/paper2/SED/weights/sed_model_base.pth`
- SED demo returns `predictions["sem_seg"]` logits/probabilities through Detectron2 `DefaultPredictor`.
- The local repo is named SED, not SED++; keep the OVIOVO backend name `sedpp` so this path can swap to a later SED++ checkout through config.
- Current OVIOVO proposal interface is `ProposalBackend.generate_proposals(rgb, depth, frame) -> list[Proposal2D]`.
- Downstream semantic memory already consumes `patch.metadata.anchor_class_name`, which is copied from `proposal.metadata.anchor_class_name` by `patch_lifting`.
- SED is semantic segmentation, not native instance segmentation. Connected components and optional depth splitting are required.
- Keep the SED++ experiment online. Do not use old proposal caches.

---

## Files

Add:

- `src/models/sedpp_proposal_backend.py`
- `configs/replica_sedpp_online_4090.yaml`
- `data/input/sedpp_replica_classes.json`
- `tests/test_sedpp_proposal_backend.py`

Modify:

- `src/modules/proposal.py`
- `src/pipelines/main_pipeline.py`
- `run_room0_full_eval.py` only if extra SED++ timing/debug fields need report support.
- `/home/ww/ww-ai/setup-oviovo0526-env.sh` only for dependency/path setup after import smoke confirms the required packages.

Docs:

- `docs/superpowers/specs/2026-06-12-sedpp-frontend-replacement-design.md`
- `docs/superpowers/plans/2026-06-12-sedpp-frontend-replacement.md`

---

## Task 1: Add SED++ Backend Factory Hook

**Files:**

- Modify: `src/modules/proposal.py`
- Add: `src/models/sedpp_proposal_backend.py`

- [x] Add `elif backend_name == "sedpp"` in `ProposalModule._create_backend()`.
- [x] Create `SEDPPProposalBackend(ProposalBackend)`.
- [x] Add a lightweight backend timing hook, for example `ProposalModule.last_generation_timings`, and merge it into pipeline frame timings from `_build_proposal_bundle()`.
- [x] Implement `initialize(config)` with config fields:
  - `repo_root`
  - `config_path`
  - `checkpoint_path`
  - `class_json`
  - `device`
  - `fast_inference`
  - `topk`
  - `min_pixel_confidence`
  - `min_component_area`
  - `max_proposals`
  - `depth_split_enabled`
- [x] Initially raise a clear error if Detectron2/SED imports fail, rather than silently falling back after partial initialization.

Expected unit check:

```bash
python -m pytest tests/test_sedpp_proposal_backend.py -q
```

---

## Task 2: Implement Semantic Component Postprocessing

**Files:**

- Modify: `src/models/sedpp_proposal_backend.py`
- Add: `tests/test_sedpp_proposal_backend.py`

- [x] Add a pure helper that converts semantic logits/probabilities to `Proposal2D` instances.
- [x] Test it without loading SED++ by passing synthetic arrays.
- [x] Use connected components per canonical class.
- [x] Drop small components.
- [x] Sort by confidence and area.
- [x] Cap to `max_proposals`.
- [x] Stamp metadata:
  - `source: sedpp_semantic_component`
  - `mask_source: sedpp_semantic_component`
  - `anchor_class_name`
  - `anchor_confidence`
  - `anchor_label_votes`
  - `anchor_label_strength`
  - `semantic_commit_allowed`
  - `sedpp_class_index`
  - `sedpp_class_name`

Precision guard test cases:

- Low-confidence component becomes unlabeled or blocked.
- Unmapped class is dropped or blocked.
- Multiple components of one class become separate proposals.
- Proposal ids are reassigned contiguously after filtering.

---

## Task 3: Add Replica-Aligned Class Vocabulary

**Files:**

- Add: `data/input/sedpp_replica_classes.json`
- Modify: `src/models/sedpp_proposal_backend.py`
- Add/modify tests as needed.

- [x] Build the first class list from Replica/OVIOVO room0 classes, not COCO-only classes.
- [x] Include common aliases through a canonicalization map in backend config.
- [x] Keep output labels canonical for Replica evaluation.
- [x] Prefer exact Replica labels when possible.

Initial canonical labels should include at least:

```text
wall, floor, ceiling, door, window, blinds, cabinet, chair, table, sofa, rug,
bed, basket, blanket, book, bottle, bowl, cushion, lamp, picture, indoor-plant,
plant-stand, pillow, shelf, stool, vase
```

Alias examples:

```text
couch -> sofa
potted plant -> indoor-plant
plant -> indoor-plant
windowpane -> window
window-blind -> blinds
floor-wood -> floor
floor-tile -> floor
wall-other -> wall
dining table -> table
coffee table -> table
```

---

## Task 4: Wire SED++ Inference

**Files:**

- Modify: `src/models/sedpp_proposal_backend.py`

- [x] Add SED repo root to `sys.path` during initialization.
- [x] Build Detectron2 config using SED's `add_sed_config`.
- [x] Load config from `/home/ww/vv/paper2/SED/configs/convnextL_768.yaml` or base variant.
- [x] Override:
  - `MODEL.WEIGHTS`
  - `MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON`
  - `MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON` if required by model construction.
  - `TEST.FAST_INFERENCE`
  - `TEST.TOPK`
  - `MODEL.DEVICE`
- [x] Run inference with RGB/BGR handling matching SED demo expectations.
- [x] Convert `predictions["sem_seg"]` into proposals through Task 2 helper.
- [x] Record backend debug info:
  - raw class count
  - kept component count
  - inference time
  - postprocess time
  - active config/weights
- [x] Expose `sedpp_inference` and `sedpp_postprocess` through the timing hook from Task 1, so 20f/200f reports can distinguish model forward time from component extraction.

If in-process imports conflict with OVIOVO dependencies, stop this task and implement Task 5 before continuing.

Current environment status: `ww-ai` has Python `3.10.20`, torch `2.6.0+cu124`, and CUDA available, but all checked container envs lack `detectron2` and importable `sed`. Explicit `backend=sedpp` initialization fails clearly with `ImportError: No module named 'detectron2'` and does not fall back to placeholder. Real backend smoke and 20f/200f validation are blocked until Detectron2/SED are installed in a compatible environment or Task 5 worker isolation is implemented.

Implementation review fixes completed:

- SED now receives a temporary plain class-list JSON even though OVIOVO stores aliases in `data/input/sedpp_replica_classes.json`.
- `sem_seg_output` is explicit and defaults to `logits`; bounded logits are no longer auto-treated as probabilities.
- Single-class logits use sigmoid instead of class-wise softmax, preventing full-frame false positives.
- Structural classes use a stricter semantic commit threshold and can emit contextual, non-committed proposals.
- `ProposalModule._finalize_proposals()` renumbers surviving proposals after common filtering.

Current targeted validation:

```bash
docker exec ww-ai bash --noprofile --norc -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python -m pytest tests/test_sedpp_proposal_backend.py tests/test_sedpp_config.py -q'
# 14 passed

docker exec ww-ai bash --noprofile --norc -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python -m pytest tests/test_pipeline.py::TestPipeline::test_pipeline_records_stage_timings_when_enabled tests/test_pipeline.py::TestPipeline::test_pipeline_merges_anchor_frontend_stage_timings tests/test_pipeline.py::TestPipeline::test_pipeline_rejects_nonfinite_nested_stage_timings_and_renames_collisions -q'
# 3 passed
```

---

## Task 5: Optional Persistent Worker Isolation

**Files:**

- Add: `src/models/sedpp_worker.py`
- Modify: `src/models/sedpp_proposal_backend.py`
- Reuse: `src/models/json_line_worker_client.py`

- [ ] Add `worker_enabled` config.
- [ ] Worker loads SED++ once and serves JSON-lines requests.
- [ ] Parent sends image path or temporary `.npy` path plus frame id.
- [ ] Worker returns compact masks or encoded components plus class/confidence/bbox.
- [ ] Parent decodes into `Proposal2D`.
- [ ] Add worker timeout and fallback behavior, but do not silently use placeholder in validation configs.

Use this if Detectron2 or SED dependency setup is incompatible with the active `ww-ai` Python environment.

---

## Task 6: Add SED++ Online Config

**Files:**

- Add: `configs/replica_sedpp_online_4090.yaml`

- [x] Start from `configs/replica_yoloworld_sam_online_baseline_4090.yaml`.
- [x] Change:

```yaml
proposal:
  backend: sedpp
  sedpp:
    device: cuda
    repo_root: /home/ww/vv/paper2/SED
    config_path: /home/ww/vv/paper2/SED/configs/convnextL_768.yaml
    checkpoint_path: /home/ww/vv/paper2/SED/weights/sed_model_large.pth
    class_json: data/input/sedpp_replica_classes.json
    fast_inference: true
    topk: 8
    min_pixel_confidence: 0.35
    min_component_area: 100
    max_proposals: 80
    depth_split_enabled: true

anchor_frontend:
  enabled: false

anchor_guided_sam:
  enabled: false

pipeline:
  yoloworld_sam_parallel_frontend_enabled: false
```

- [x] Keep downstream `runtime_vis`, `depth_refinement`, `association`, and `object_update` close to the current baseline for clean A/B comparison.

Config/class-table validation completed:

```bash
docker exec ww-ai bash --noprofile --norc -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && python -m pytest tests/test_sedpp_config.py -q'
```

---

## Task 7: Backend Smoke Test In Container

**Command:**

```bash
docker exec ww-ai bash --noprofile --norc -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
python - <<'"'"'PY'"'"'
from pathlib import Path
import numpy as np
from src.models.sedpp_proposal_backend import SEDPPProposalBackend

backend = SEDPPProposalBackend()
backend.initialize({
    "device": "cuda",
    "repo_root": "/home/ww/vv/paper2/SED",
    "config_path": "/home/ww/vv/paper2/SED/configs/convnextL_768.yaml",
    "checkpoint_path": "/home/ww/vv/paper2/SED/weights/sed_model_large.pth",
    "class_json": "data/input/sedpp_replica_classes.json",
    "fast_inference": True,
    "topk": 8,
    "min_pixel_confidence": 0.35,
    "min_component_area": 100,
    "max_proposals": 80,
})
rgb = np.zeros((480, 640, 3), dtype=np.uint8)
depth = np.ones((480, 640), dtype=np.float32)
props = backend.generate_proposals(rgb, depth)
print({"count": len(props), "debug": backend.last_generation_info})
PY
'
```

Expected:

- SED++ imports.
- Weights load.
- No placeholder fallback.
- A real RGB frame should return non-empty proposals before moving to eval.

---

## Task 8: 20f Validation

**Command:**

```bash
docker exec ww-ai bash --noprofile --norc -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
experiment=20260612_sedpp_room0_s10_20f_smoke
log=outputs/tmp_validation/${experiment}.log
status=outputs/tmp_validation/${experiment}.status
timefile=outputs/tmp_validation/${experiment}.time.txt
rm -f "$log" "$status" "$timefile"
/usr/bin/time -v python run_room0_full_eval.py \
  --config-path configs/replica_sedpp_online_4090.yaml \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --output-root outputs/tmp_validation \
  --experiment-name "$experiment" \
  --num-frames 20 \
  --frame-stride 10 \
  --proposal-backend sedpp \
  --fast-eval > "$log" 2> >(tee "$timefile" >&2)
code=$?
echo "$code" > "$status"
exit "$code"
'
```

Checks:

- `status` is `0`.
- `frame_metrics.jsonl` has 20 lines.
- `stage_timings.proposal_generation` exists.
- `proposal_source` is not `anchor_guided_sam`.
- proposal backend is `sedpp`.
- Mean raw proposal count is not zero and not exploding.
- Semantic labels are present in lifted patches.

---

## Task 9: 200f A/B Validation

Run only after 20f smoke passes.

**Command:**

```bash
docker exec ww-ai bash --noprofile --norc -lc '
cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
experiment=20260612_sedpp_room0_s10_200f
log=outputs/tmp_validation/${experiment}.log
status=outputs/tmp_validation/${experiment}.status
timefile=outputs/tmp_validation/${experiment}.time.txt
rm -f "$log" "$status" "$timefile"
/usr/bin/time -v python run_room0_full_eval.py \
  --config-path configs/replica_sedpp_online_4090.yaml \
  --dataset-root /home/ww/vv/dataset/Replica/room0 \
  --gt-labels data/input/replica_semantic_gt/room0.txt \
  --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply \
  --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json \
  --output-root outputs/tmp_validation \
  --experiment-name "$experiment" \
  --num-frames 200 \
  --frame-stride 10 \
  --proposal-backend sedpp \
  --fast-eval > "$log" 2> >(tee "$timefile" >&2)
code=$?
echo "$code" > "$status"
exit "$code"
'
```

Decision table:

| metric | current YOLO+SAM | SED++ target | minimum viable |
| --- | ---: | ---: | ---: |
| mIoU | `0.5190` | `>= 0.49` | `>= 0.47` |
| f-mIoU | `0.6052` | `>= 0.58` | `>= 0.56` |
| proposal_generation mean | `10.4371s` | `<= 2.5s` | `<= 4.0s` |
| wall time | `40:37.90` | `<= 22min` | `<= 30min` |

---

## Task 10: If Accuracy Drops Too Much

Apply these in order, one change per experiment:

- [ ] Enable depth split for large semantic components.
- [ ] Increase confidence threshold for structural classes only.
- [ ] Add class-specific area caps for large surfaces.
- [ ] Add low-confidence unlabeled residuals with `semantic_commit_allowed: false`.
- [ ] Add sparse SAM2 rescue for compact low-confidence object regions only.
- [ ] Add sparse YOLOWorld semantic audit every `N` frames only if SED++ class labels are unstable.

Do not reintroduce full per-frame YOLOWorld + SAM2 + anchor-guided fusion unless comparing against the baseline.

---

## Final Report Template

Create a short report under `docs/superpowers/reports/` after 200f validation:

```text
# SED++ Frontend Room0 Results

- Run:
- Config:
- Frames:
- Stride:
- mIoU:
- f-mIoU:
- Wall time:
- proposal_generation mean:
- sedpp_inference mean:
- sedpp_postprocess mean:
- mean proposal count:
- final object count:

Decision:
- Keep / tune / reject.

Notes:
- Accuracy regressions by class:
- Timing bottlenecks:
- Next tuning step:
```
