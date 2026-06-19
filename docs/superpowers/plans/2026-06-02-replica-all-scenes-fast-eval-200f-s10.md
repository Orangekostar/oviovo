# Replica All-Scenes Fast-Eval 200f Stride10 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a comparable Replica multi-scene experiment where each available scene is processed with `200f / stride=10` using the current fast-eval checkpointed flow. Use one fixed global open-vocabulary word list and report per-scene plus aggregate metrics without rerunning mapping for export/eval.

**Architecture:** Keep the current `fast_eval` output profile and checkpointed runner behavior: mapping writes `mapping_state.pkl`, `mapping_timer_result.json`, `frame_metrics.jsonl`; export/eval reuses the saved state to write `run_report.md`, `run_report.json`, `results.json`, and primary PLYs. Add scene parametrization and a global Replica vocabulary manifest. Do not enable full debug artifacts for the all-scene run.

**Tech Stack:** Python, argparse, YAML/JSON vocab manifests, NumPy, pytest, existing `run_room0_full_eval.py` export helpers, existing `scripts/run_room0_checkpointed_eval.py`, Replica dataset paths under `/home/ww/vv/dataset`.

---

## Evidence And Constraints

- Current validated checkpointed room0 run:
  - `outputs/tmp_validation/20260602_room0_checkpointed_s10_200f_gpu`
  - `processed_frames=200`
  - `frame_stride=10`
  - `mapping_loop_sec_excluding_init_and_final_outputs=1931.4897`
  - `export_eval_sec=10.0362`
  - `mIoU=0.5637`, `f-mIoU=0.7244`, `mAcc=0.6483`, `f-mAcc=0.8285`
  - `prefetch_hit_count=200/200`
- Use `--fast-eval` for every scene:
  - Keep primary exports, metrics, final semantic audit, reports.
  - Skip local-memory audit, debug duplicate PLY, GT mesh eval PLYs, and preview PNGs.
- Current local Replica trajectory dataset has these available scene dirs:
  - `/home/ww/vv/dataset/Replica/room0`
  - `/home/ww/vv/dataset/Replica/room1`
  - `/home/ww/vv/dataset/Replica/room2`
  - `/home/ww/vv/dataset/Replica/office0`
  - `/home/ww/vv/dataset/Replica/office1`
  - `/home/ww/vv/dataset/Replica/office2`
  - `/home/ww/vv/dataset/Replica/office3`
  - `/home/ww/vv/dataset/Replica/office4`
- Current local GT label files exist for those same eight scenes:
  - `data/input/replica_semantic_gt/{room0,room1,room2,office0,office1,office2,office3,office4}.txt`
- Ground-truth mesh/info scene-name mapping:
  - `room0 -> room_0`
  - `room1 -> room_1`
  - `room2 -> room_2`
  - `office0 -> office_0`
  - `office1 -> office_1`
  - `office2 -> office_2`
  - `office3 -> office_3`
  - `office4 -> office_4`
- Full Replica original dataset also includes apartments/hotel/frl apartments, but matching local RGB-D trajectory folders and label txt files are not currently present. This plan treats the eight available RGB-D scenes as the runnable all-scene set unless those missing paths are added later.
- Do not compare scenes with different prompt lists. Use one global canonical vocabulary and one fixed alias mapping.

---

## Vocabulary Design Contract

The experiment uses three related vocabularies:

1. **Canonical eval vocabulary**
   - Union of official Replica class names from the eight selected scenes' `info_semantic.json`.
   - Used for reporting, alias targets, and final canonical class names.
   - Must not vary by scene.

2. **Prompt alias vocabulary**
   - A conservative alias list mapped back to canonical names.
   - Examples:
     - `sofa`: `sofa`, `couch`
     - `rug`: `rug`, `carpet`
     - `picture`: `picture`, `painting`, `wall art`
     - `wall-plug`: `wall plug`, `outlet`, `electrical outlet`
     - `indoor-plant`: `indoor plant`, `plant`
     - `cabinet`: `cabinet`, `cupboard`
     - `blinds`: `blinds`, `window blinds`
   - Aliases must never become independent eval classes.

3. **Structural/stuff vocabulary**
   - Keep structural classes separate from object-pool semantics when possible:
     - `wall`
     - `floor`
     - `ceiling`
     - `door`
     - `window`
     - `blinds`
     - `rug`
     - `pillar`
     - `vent`
   - Remaining canonical classes are object labels.

Do not add broad prompts such as `object`, `thing`, `furniture`, `decor`, or `background`.

---

## File Structure

- Add: `scripts/build_replica_global_vocab.py`
  - Scan selected scene `info_semantic.json` files.
  - Write `configs/replica_global_vocab.yaml`.
  - Write `outputs/tmp_validation/replica_global_vocab_summary.json`.

- Add: `configs/replica_global_vocab.yaml`
  - `canonical_eval_vocab`
  - `prompt_aliases`
  - `structural_labels`
  - `object_labels`
  - `scene_class_coverage`

- Modify: `scripts/run_room0_checkpointed_eval.py`
  - Generalize scene naming where needed, or keep the existing room0 implementation as the underlying single-scene runner only if all paths can be passed explicitly.
  - Required scene args:
    - `--scene-name`
    - `--dataset-root`
    - `--gt-labels`
    - `--gt-mesh-ply`
    - `--gt-info-json`
  - Ensure output layout uses the scene name instead of hard-coded `room0` for scene subdir and file prefixes if multi-scene artifacts need clear naming.

- Add: `scripts/run_replica_all_scenes_fast_eval.py`
  - Iterate selected scenes.
  - Call the checkpointed runner with:
    - `--mode build-export`
    - `--num-frames 200`
    - `--frame-stride 10`
    - `--fast-eval`
    - `--proposal-backend precomputed`
    - `--proposal-device cuda`
    - `--quiet`
  - Write one run root per scene.
  - Skip completed scenes only when `status.json` says `complete`, `frame_metrics.jsonl` has 200 lines, and `results.json` exists.

- Add: `scripts/summarize_replica_all_scenes.py`
  - Read per-scene `mapping_timer_result.json`, `export_eval_timer_result.json`, `replica/results.json`, and `room0/run_report.json` or generalized scene report path.
  - Write:
    - `outputs/tmp_validation/<batch_name>/summary.json`
    - `outputs/tmp_validation/<batch_name>/summary.md`

- Modify: `tests/test_dual_map.py` or add `tests/test_replica_all_scenes.py`
  - Unit-test scene path mapping.
  - Unit-test vocab generation on temporary `info_semantic.json` files.
  - Unit-test summary aggregation on small fake output dirs.

---

## Task 1: Generate The Global Replica Vocabulary

**Files:**
- Add: `scripts/build_replica_global_vocab.py`
- Add: `configs/replica_global_vocab.yaml`
- Add/Modify tests.

- [ ] **Step 1: Write failing tests for vocabulary generation**

Create tests that build two temporary `info_semantic.json` files and assert:

- canonical class names are unioned and sorted.
- aliases map only to canonical names.
- structural labels are separated.
- object labels exclude structural labels.
- per-scene coverage is recorded.

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py -q
```

Expected: FAIL because the script/helper does not exist.

- [ ] **Step 2: Implement vocabulary builder**

Implement `scripts/build_replica_global_vocab.py` with a pure helper such as:

```python
def build_replica_vocab(scene_info_paths: dict[str, Path]) -> dict[str, object]:
    ...
```

The output YAML must include:

```yaml
canonical_eval_vocab: [...]
prompt_aliases:
  sofa: [sofa, couch]
structural_labels: [...]
object_labels: [...]
scene_class_coverage:
  room0: [...]
```

- [ ] **Step 3: Generate the real vocab file**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/build_replica_global_vocab.py \
  --output configs/replica_global_vocab.yaml \
  --summary-output outputs/tmp_validation/replica_global_vocab_summary.json
```

- [ ] **Step 4: Validate**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest tests/test_replica_all_scenes.py -q
```

Expected: PASS.

---

## Task 2: Generalize The Checkpointed Runner For Scene Names

**Files:**
- Modify: `scripts/run_room0_checkpointed_eval.py`
- Modify: `run_room0_full_eval.py` only if layout/export helpers need scene-name parametrization.
- Add/Modify tests.

- [ ] **Step 1: Add failing tests for scene layout**

Assert that a generalized layout for `office0` writes scene artifacts under `office0/` rather than `room0/`, and uses the passed GT paths.

- [ ] **Step 2: Add scene args**

Add:

```bash
--scene-name
```

Default can remain `room0` for backward compatibility.

- [ ] **Step 3: Remove hard-coded room0 assumptions where required**

Keep existing filenames acceptable if downstream tools expect them, but the scene directory should not collide across scenes. Preferred:

```text
<run_root>/<scene_name>/mapping_state.pkl
<run_root>/<scene_name>/run_report.md
<run_root>/<scene_name>/exports/...
<run_root>/replica/results.json
```

If a deeper refactor is too risky, keep the existing `room0/` subdir inside each scene-specific run root and document it. The batch runner must still produce unique run roots per scene.

- [ ] **Step 4: Focused smoke**

Run one non-room scene with `--num-frames 1 --frame-stride 10 --fast-eval --quiet`, for example `room1` or `office0`.

Expected artifacts:

- `mapping_state.pkl`
- `frame_metrics.jsonl`
- `run_report.md`
- `replica/results.json`
- primary PLY exports

---

## Task 3: Add The All-Scene Fast-Eval Batch Runner

**Files:**
- Add: `scripts/run_replica_all_scenes_fast_eval.py`
- Add/Modify tests.

- [ ] **Step 1: Define selected scenes**

Use this default scene list:

```python
SCENES = {
    "room0": "room_0",
    "room1": "room_1",
    "room2": "room_2",
    "office0": "office_0",
    "office1": "office_1",
    "office2": "office_2",
    "office3": "office_3",
    "office4": "office_4",
}
```

- [ ] **Step 2: Build per-scene paths**

For each scene:

```text
dataset_root = /home/ww/vv/dataset/Replica/<scene>
gt_labels = data/input/replica_semantic_gt/<scene>.txt
gt_mesh_ply = /home/ww/vv/dataset/Replica-Dataset/Replica_original/<gt_scene>/habitat/mesh_semantic.ply
gt_info_json = /home/ww/vv/dataset/Replica-Dataset/Replica_original/<gt_scene>/habitat/info_semantic.json
```

- [ ] **Step 3: Implement completion checks**

A scene is complete only if all are true:

- `status.json` has `"status": "complete"`.
- `frame_metrics.jsonl` has 200 lines.
- `mapping_state.pkl` exists.
- `replica/results.json` exists.
- `room0/run_report.md` or generalized scene report exists.

- [ ] **Step 4: Run commands sequentially by default**

Do not run multiple scenes concurrently on one GPU by default. Each scene should call:

```bash
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_room0_checkpointed_eval.py \
  --mode build-export \
  --experiment-name <batch_name>_<scene>_s10_200f_fast_eval \
  --dataset-root <dataset_root> \
  --gt-labels <gt_labels> \
  --gt-mesh-ply <gt_mesh_ply> \
  --gt-info-json <gt_info_json> \
  --config-path configs/room0_surface_gate_fast_high_iou_4090.yaml \
  --num-frames 200 \
  --frame-stride 10 \
  --fast-eval \
  --proposal-backend precomputed \
  --proposal-device cuda \
  --quiet
```

- [ ] **Step 5: Add dry-run mode**

`--dry-run` should print all per-scene commands and validate all input paths without running mapping.

---

## Task 4: Add Batch Summary

**Files:**
- Add: `scripts/summarize_replica_all_scenes.py`
- Add/Modify tests.

- [ ] **Step 1: Write fake-output aggregation tests**

Create temporary scene output dirs containing minimal:

- `mapping_timer_result.json`
- `export_eval_timer_result.json`
- `replica/results.json`
- `room0/run_report.json`

Assert summary contains:

- per-scene metrics
- macro-average mIoU/f-mIoU/mAcc/f-mAcc
- mean TPF
- total mapping time
- total export/eval time
- object count stats

- [ ] **Step 2: Implement summarizer**

Write both JSON and Markdown summary.

Markdown table columns:

```text
scene | mIoU | f-mIoU | mAcc | f-mAcc | TPF | mapping_sec | export_sec | objects | dense_points | prefetch
```

---

## Task 5: Smoke Validation

**Files:** no code edits expected after this task unless smoke fails.

- [ ] **Step 1: Build global vocab**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/build_replica_global_vocab.py \
  --output configs/replica_global_vocab.yaml \
  --summary-output outputs/tmp_validation/replica_global_vocab_summary.json
```

- [ ] **Step 2: Dry-run all-scene commands**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_replica_all_scenes_fast_eval.py \
  --batch-name 20260602_replica_all_scenes_s10_200f_fast_eval \
  --dry-run
```

Expected:

- eight selected scenes
- all paths valid
- commands include `--fast-eval --num-frames 200 --frame-stride 10`

- [ ] **Step 3: One-scene smoke**

Run one quick smoke:

```bash
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_replica_all_scenes_fast_eval.py \
  --batch-name 20260602_replica_all_scenes_smoke \
  --scenes room1 \
  --num-frames 1 \
  --frame-stride 10
```

Expected:

- status complete
- checkpoint exists
- results/report/PLY exists

---

## Task 6: Full All-Scene Run

**Files:** runtime outputs only.

- [ ] **Step 1: Run all selected scenes**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 \
/home/ww/miniconda3/envs/oviovo/bin/python scripts/run_replica_all_scenes_fast_eval.py \
  --batch-name 20260602_replica_all_scenes_s10_200f_fast_eval \
  --num-frames 200 \
  --frame-stride 10
```

- [ ] **Step 2: Summarize**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python scripts/summarize_replica_all_scenes.py \
  --batch-name 20260602_replica_all_scenes_s10_200f_fast_eval
```

- [ ] **Step 3: Report final results**

Report:

- per-scene metrics table
- macro averages
- total wall/mapping/export time
- failed/skipped scenes, if any
- output paths

---

## Expected Outputs

For each scene:

```text
outputs/tmp_validation/<batch_name>_<scene>_s10_200f_fast_eval/
  status.json
  mapping_timer_result.json
  export_eval_timer_result.json
  silent_run.log
  room0/ or <scene>/
    mapping_state.pkl
    frame_metrics.jsonl
    run_report.md
    run_report.json
    final_object_semantic_audit.json
    final_object_semantic_audit.md
    exports/
      room0_instance_map.ply
      room0_instance_map_dense_surface.ply
      room0_instance_map_tsdf_backbone.ply
      room0_dense_geometry_fused_rgb.ply
      room0_dense_geometry_instance_projected.ply
      room0_structural_overlay.ply
  replica/
    results.json
    classes_iou.json
    classes_acc.json
    statistics.txt
```

Batch summary:

```text
outputs/tmp_validation/20260602_replica_all_scenes_s10_200f_fast_eval/
  summary.json
  summary.md
```

---

## Risks And Mitigations

- **Risk: precomputed proposal caches may only cover room0.**
  - Mitigation: smoke one non-room scene before the full run. If cache is missing, either generate cache or switch backend explicitly and record the changed runtime contract.

- **Risk: current runner names everything room0.**
  - Mitigation: unique run roots prevent collision. Generalize scene subdirs only if needed for clarity; do not block the experiment on cosmetic filenames.

- **Risk: all-scene run is long.**
  - Mitigation: sequential runner resumes completed scenes using status checks. Checkpointed export avoids repeating mapping when only summary/export needs rerun.

- **Risk: global alias list adds false positives.**
  - Mitigation: keep aliases conservative and map aliases to canonical labels. Do not add broad umbrella prompts.

- **Risk: missing GT labels for full original Replica scenes.**
  - Mitigation: declare the runnable set as the eight local RGB-D scenes with matching label txt files. Add apartments/hotel/frl apartments only after matching RGB-D and label files are available.

---

## Completion Criteria

- `configs/replica_global_vocab.yaml` exists and includes canonical/alias/object/structural sections.
- Dry-run validates all eight selected scenes.
- At least one non-room scene smoke passes with `--fast-eval`.
- Full all-scene run completes or reports explicit failed scenes.
- `summary.md` and `summary.json` contain per-scene metrics and macro averages.
- No scene reruns mapping just to regenerate eval/report/PLY.
