# Room0 Fast Eval Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast evaluation output mode for the best 2026-05-26 observation-first room0 workflow that preserves algorithm behavior while skipping heavy debug artifacts.

**Architecture:** Keep the pipeline, proposal, association, object update, semantic, and evaluation logic unchanged. Add an explicit runner output profile in `run_room0_full_eval.py` so `--fast-eval` disables local-memory audits, debug duplicate PLYs, GT mesh visualization PLYs, and dense preview PNGs while retaining metrics, final semantic audit, run reports, and the primary instance PLY needed for comparison.

**Tech Stack:** Python, argparse, dataclasses, NumPy, pytest, existing OVIOVO room0 runner.

---

## File Structure

- Modify: `run_room0_full_eval.py`
  - Add `OutputProfile` and `build_output_profile(args)`.
  - Add CLI flags:
    - `--fast-eval`
    - `--full-debug`
    - `--skip-debug-ply`
    - `--skip-eval-ply`
    - `--skip-preview-render`
  - Route artifact writes through the output profile without changing runtime pipeline behavior.
  - Make `write_eval_artifacts()` optionally skip GT mesh PLYs but always write JSON metrics.
  - Make `write_run_report()` report output mode and only list files actually written.

- Modify: `tests/test_dual_map.py`
  - Add tests for output profile flag behavior.
  - Add tests for `write_eval_artifacts(write_mesh_ply=False)`.
  - Update run-report tests to verify skipped debug exports are not listed.

- Runtime validation only:
  - Run a 0526-style `room0` smoke with `--fast-eval --num-frames 20 --frame-stride 10` and precomputed proposals.
  - Confirm no `local_memory_audit/`, no dense preview PNGs, no debug duplicate PLY, and no GT mesh semantic PLYs are produced.

---

## Design Contract

The fast mode must not affect these execution paths:

- `Pipeline.process_frame(...)`
- frontend proposal loading and anchor assignment
- runtime merge/refinement/lifting
- association and contested residual routing
- object update / TSDF / dense surface updates
- semantic label selection
- `evaluate_semantics(...)`

The fast mode changes only file-system output after or around these computations.

The 0526 best-effect run was `20260526_room0_observation_first_fix52e1714_200f`, based on commit `52e1714` / `2b830e1` behavior. This plan is safe to execute on the current worktree because it only changes output gating in the runner. If the user wants a historical branch later, cherry-pick these runner changes onto a worktree based at `52e1714` or `2b830e1`.

---

### Task 1: Add Output Profile Unit Contract

**Files:**
- Modify: `tests/test_dual_map.py`
- Modify: `run_room0_full_eval.py`

- [ ] **Step 1: Add imports for the output profile tests**

In `tests/test_dual_map.py`, update the existing import from `run_room0_full_eval` to include:

```python
    build_output_profile,
    write_eval_artifacts,
```

The import block should include both names alongside existing imports such as `write_run_report`.

- [ ] **Step 2: Add failing tests for output profile modes**

Append these tests after `test_eval_class_colors_share_semantic_palette_with_exports`:

```python
def test_fast_eval_output_profile_disables_heavy_debug_artifacts() -> None:
    args = SimpleNamespace(
        fast_eval=True,
        full_debug=False,
        lightweight_benchmark=False,
        skip_debug_ply=False,
        skip_eval_ply=False,
        skip_preview_render=False,
    )

    profile = build_output_profile(args)

    assert profile.name == "fast_eval"
    assert profile.benchmark_audit_enabled is False
    assert profile.write_debug_ply is False
    assert profile.write_eval_mesh_ply is False
    assert profile.write_preview_render is False
    assert profile.write_primary_exports is True
    assert profile.write_final_audit is True
    assert profile.write_reports is True


def test_full_debug_output_profile_keeps_existing_artifacts() -> None:
    args = SimpleNamespace(
        fast_eval=True,
        full_debug=True,
        lightweight_benchmark=True,
        skip_debug_ply=True,
        skip_eval_ply=True,
        skip_preview_render=True,
    )

    profile = build_output_profile(args)

    assert profile.name == "full_debug"
    assert profile.benchmark_audit_enabled is True
    assert profile.write_debug_ply is True
    assert profile.write_eval_mesh_ply is True
    assert profile.write_preview_render is True
    assert profile.write_primary_exports is True
    assert profile.write_final_audit is True
    assert profile.write_reports is True


def test_selective_skip_flags_disable_individual_artifacts() -> None:
    args = SimpleNamespace(
        fast_eval=False,
        full_debug=False,
        lightweight_benchmark=False,
        skip_debug_ply=True,
        skip_eval_ply=True,
        skip_preview_render=True,
    )

    profile = build_output_profile(args)

    assert profile.name == "custom"
    assert profile.benchmark_audit_enabled is True
    assert profile.write_debug_ply is False
    assert profile.write_eval_mesh_ply is False
    assert profile.write_preview_render is False
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_dual_map.py::test_fast_eval_output_profile_disables_heavy_debug_artifacts \
  tests/test_dual_map.py::test_full_debug_output_profile_keeps_existing_artifacts \
  tests/test_dual_map.py::test_selective_skip_flags_disable_individual_artifacts \
  -q
```

Expected: FAIL with `ImportError` or `AttributeError` because `build_output_profile` is not implemented.

- [ ] **Step 4: Add the output profile implementation**

In `run_room0_full_eval.py`, add `dataclass` to the imports:

```python
from dataclasses import dataclass
```

Add this dataclass after `PLY_DTYPE = np.dtype(...)`:

```python
@dataclass(frozen=True)
class OutputProfile:
    name: str
    benchmark_audit_enabled: bool
    write_primary_exports: bool = True
    write_debug_ply: bool = True
    write_eval_mesh_ply: bool = True
    write_preview_render: bool = True
    write_final_audit: bool = True
    write_reports: bool = True
```

Add this function before `parse_args()`:

```python
def build_output_profile(args: argparse.Namespace) -> OutputProfile:
    """Build the output artifact profile without changing pipeline behavior."""
    if bool(getattr(args, "full_debug", False)):
        return OutputProfile(
            name="full_debug",
            benchmark_audit_enabled=True,
            write_primary_exports=True,
            write_debug_ply=True,
            write_eval_mesh_ply=True,
            write_preview_render=True,
            write_final_audit=True,
            write_reports=True,
        )

    if bool(getattr(args, "fast_eval", False)):
        return OutputProfile(
            name="fast_eval",
            benchmark_audit_enabled=False,
            write_primary_exports=True,
            write_debug_ply=False,
            write_eval_mesh_ply=False,
            write_preview_render=False,
            write_final_audit=True,
            write_reports=True,
        )

    benchmark_audit_enabled = not bool(getattr(args, "lightweight_benchmark", False))
    debug_ply_enabled = not bool(getattr(args, "skip_debug_ply", False))
    eval_mesh_ply_enabled = not bool(getattr(args, "skip_eval_ply", False))
    preview_render_enabled = not bool(getattr(args, "skip_preview_render", False))
    name = (
        "custom"
        if (
            bool(getattr(args, "skip_debug_ply", False))
            or bool(getattr(args, "skip_eval_ply", False))
            or bool(getattr(args, "skip_preview_render", False))
            or bool(getattr(args, "lightweight_benchmark", False))
        )
        else "full_debug"
    )
    return OutputProfile(
        name=name,
        benchmark_audit_enabled=benchmark_audit_enabled,
        write_primary_exports=True,
        write_debug_ply=debug_ply_enabled,
        write_eval_mesh_ply=eval_mesh_ply_enabled,
        write_preview_render=preview_render_enabled,
        write_final_audit=True,
        write_reports=True,
    )
```

- [ ] **Step 5: Add CLI flags**

In `parse_args()`, after the existing `--lightweight-benchmark` argument, add:

```python
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help=(
            "Use fast evaluation outputs: keep metrics, primary exports, final semantic audit, "
            "and reports; skip local-memory audit, debug duplicate PLYs, GT mesh PLYs, and previews."
        ),
    )
    parser.add_argument(
        "--full-debug",
        action="store_true",
        help="Force all debug artifacts on, overriding --fast-eval and skip flags.",
    )
    parser.add_argument(
        "--skip-debug-ply",
        action="store_true",
        help="Skip debug-only duplicate PLY exports such as local-memory compatibility PLY.",
    )
    parser.add_argument(
        "--skip-eval-ply",
        action="store_true",
        help="Skip GT mesh semantic visualization PLYs while keeping JSON evaluation metrics.",
    )
    parser.add_argument(
        "--skip-preview-render",
        action="store_true",
        help="Skip dense projection preview PNG rendering.",
    )
```

- [ ] **Step 6: Run output profile tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_dual_map.py::test_fast_eval_output_profile_disables_heavy_debug_artifacts \
  tests/test_dual_map.py::test_full_debug_output_profile_keeps_existing_artifacts \
  tests/test_dual_map.py::test_selective_skip_flags_disable_individual_artifacts \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add run_room0_full_eval.py tests/test_dual_map.py
git commit -m "feat: add room0 output profiles"
```

---

### Task 2: Gate Runner Artifacts Through Output Profile

**Files:**
- Modify: `run_room0_full_eval.py`
- Modify: `tests/test_dual_map.py`

- [ ] **Step 1: Add eval artifact skip test**

Append this test after the output profile tests in `tests/test_dual_map.py`:

```python
def test_write_eval_artifacts_can_skip_mesh_plys(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    gt_vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    gt_labels = np.array([1, 2], dtype=np.int32)
    evaluation = {
        "miou": 0.25,
        "macc": 0.50,
        "fmiou": 0.75,
        "fmacc": 1.0,
        "pred_labels": np.array([1, 1], dtype=np.int32),
        "correctness_colors": np.array([[0, 255, 0], [255, 0, 0]], dtype=np.uint8),
        "class_ious": {"chair": 0.25},
        "class_accs": {"chair": 0.50},
        "per_class": [{"class_name": "chair", "acc": 0.50, "iou": 0.25}],
    }

    write_eval_artifacts(
        eval_dir,
        gt_vertices,
        gt_labels,
        evaluation,
        {1: "chair", 2: "table"},
        write_mesh_ply=False,
    )

    assert (eval_dir / "results.json").exists()
    assert (eval_dir / "classes_iou.json").exists()
    assert (eval_dir / "classes_acc.json").exists()
    assert (eval_dir / "statistics.txt").exists()
    assert not (eval_dir / "room0_gtmesh_gt_semantic.ply").exists()
    assert not (eval_dir / "room0_gtmesh_pred_semantic.ply").exists()
    assert not (eval_dir / "room0_gtmesh_semantic_correctness.ply").exists()
```

- [ ] **Step 2: Run eval artifact test and verify it fails**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_dual_map.py::test_write_eval_artifacts_can_skip_mesh_plys \
  -q
```

Expected: FAIL with `TypeError: write_eval_artifacts() got an unexpected keyword argument 'write_mesh_ply'`.

- [ ] **Step 3: Update `write_eval_artifacts()` signature and body**

In `run_room0_full_eval.py`, change the signature to:

```python
def write_eval_artifacts(
    eval_dir: Path,
    gt_vertices: np.ndarray,
    gt_labels: np.ndarray,
    evaluation: dict[str, Any],
    class_names: dict[int, str],
    *,
    write_mesh_ply: bool = True,
) -> None:
```

Wrap the mesh PLY color computation and `write_vertex_ply(...)` calls in:

```python
    if write_mesh_ply:
        gt_colors = labels_to_class_colors(gt_labels, class_names=class_names)
        pred_colors = labels_to_class_colors(pred_labels, class_names=class_names)

        write_vertex_ply(
            eval_dir / "room0_gtmesh_gt_semantic.ply",
            gt_vertices,
            gt_colors,
            gt_labels.astype(np.int32),
            np.zeros(len(gt_vertices), dtype=np.uint8),
            np.zeros(len(gt_vertices), dtype=np.float32),
        )
        write_vertex_ply(
            eval_dir / "room0_gtmesh_pred_semantic.ply",
            gt_vertices,
            pred_colors,
            pred_labels.astype(np.int32),
            np.zeros(len(gt_vertices), dtype=np.uint8),
            np.zeros(len(gt_vertices), dtype=np.float32),
        )
        write_vertex_ply(
            eval_dir / "room0_gtmesh_semantic_correctness.ply",
            gt_vertices,
            correctness,
            pred_labels.astype(np.int32),
            np.zeros(len(gt_vertices), dtype=np.uint8),
            np.zeros(len(gt_vertices), dtype=np.float32),
        )
```

Leave JSON and `statistics.txt` writes outside the `if` block.

- [ ] **Step 4: Use output profile in `main()`**

In `main()`, replace:

```python
    benchmark_audit_enabled = not bool(args.lightweight_benchmark)
```

with:

```python
    output_profile = build_output_profile(args)
    benchmark_audit_enabled = bool(output_profile.benchmark_audit_enabled)
```

After:

```python
    print(f"benchmark_audit_enabled: {benchmark_audit_enabled}")
```

add:

```python
    print(f"output_profile: {output_profile.name}")
```

- [ ] **Step 5: Gate debug duplicate PLY write**

Replace:

```python
    local_memory_path = exports_dir / "room0_instance_map_local_memory.ply"
    write_binary_ply(instance_map_path, pool_semantic_records)
    write_binary_ply(dense_surface_path, dense_surface_records)
    write_binary_ply(instance_backbone_path, tsdf_records)
    write_binary_ply(local_memory_path, pool_debug_records)
```

with:

```python
    local_memory_path = exports_dir / "room0_instance_map_local_memory.ply"
    write_binary_ply(instance_map_path, pool_semantic_records)
    write_binary_ply(dense_surface_path, dense_surface_records)
    write_binary_ply(instance_backbone_path, tsdf_records)
    if output_profile.write_debug_ply:
        write_binary_ply(local_memory_path, pool_debug_records)
```

- [ ] **Step 6: Gate preview PNG rendering**

Wrap the four `render_projection(...)` calls and `render_preview_sheet(...)` call in:

```python
    if output_profile.write_preview_render:
        render_projection(dense_points, dense_rgb, plane="xz", output_path=vis_dir / "room0_dense_xz_rgb.png")
        render_projection(dense_points, projected_colors, plane="xz", output_path=vis_dir / "room0_dense_xz_instance.png")
        render_projection(dense_points, dense_rgb, plane="xy", output_path=vis_dir / "room0_dense_xy_rgb.png")
        render_projection(dense_points, projected_colors, plane="xy", output_path=vis_dir / "room0_dense_xy_instance.png")
        render_preview_sheet(
            [
                vis_dir / "room0_dense_xz_rgb.png",
                vis_dir / "room0_dense_xz_instance.png",
                vis_dir / "room0_dense_xy_rgb.png",
                vis_dir / "room0_dense_xy_instance.png",
            ],
            vis_dir / "room0_dense_projection_preview.png",
        )
```

- [ ] **Step 7: Gate eval mesh PLYs**

Replace:

```python
    write_eval_artifacts(eval_dir, gt_vertices, gt_labels, evaluation, class_names)
```

with:

```python
    write_eval_artifacts(
        eval_dir,
        gt_vertices,
        gt_labels,
        evaluation,
        class_names,
        write_mesh_ply=output_profile.write_eval_mesh_ply,
    )
```

- [ ] **Step 8: Pass output profile to run report**

Change the `write_run_report(...)` call to add:

```python
        output_profile=output_profile,
```

Add this parameter to the function signature:

```python
    output_profile: OutputProfile | None = None,
```

At the start of `write_run_report()`, after `final_object_semantic_audit = final_object_semantic_audit or {}`, add:

```python
    output_profile = output_profile or OutputProfile(
        name="full_debug",
        benchmark_audit_enabled=audit_dir is not None,
    )
```

- [ ] **Step 9: Update report file list to match actual outputs**

In `write_run_report()`, before building `report`, add:

```python
    export_lines = [
        f"- `room0_instance_map.ply` (pool-based semantic object map): `{scene_dir / 'exports' / 'room0_instance_map.ply'}`",
        f"- `room0_instance_map_dense_surface.ply` (resident dense surface export): `{scene_dir / 'exports' / 'room0_instance_map_dense_surface.ply'}`",
        f"- `room0_instance_map_tsdf_backbone.ply` (coarse TSDF owner/support/stability backbone): `{scene_dir / 'exports' / 'room0_instance_map_tsdf_backbone.ply'}`",
        f"- `room0_dense_geometry_instance_projected.ply` (projected from the pool-based instance map): `{scene_dir / 'exports' / 'room0_dense_geometry_instance_projected.ply'}`",
    ]
    if output_profile.write_debug_ply:
        export_lines.append(
            f"- `room0_instance_map_local_memory.ply` (compat/debug pool view): `{scene_dir / 'exports' / 'room0_instance_map_local_memory.ply'}`"
        )
    if output_profile.write_eval_mesh_ply:
        export_lines.extend(
            [
                f"- `room0_gtmesh_pred_semantic.ply`: `{eval_dir / 'room0_gtmesh_pred_semantic.ply'}`",
                f"- `room0_gtmesh_gt_semantic.ply`: `{eval_dir / 'room0_gtmesh_gt_semantic.ply'}`",
                f"- `room0_gtmesh_semantic_correctness.ply`: `{eval_dir / 'room0_gtmesh_semantic_correctness.ply'}`",
            ]
        )
    export_lines.extend(
        [
            f"- `final_object_semantic_audit.json`: `{scene_dir / 'final_object_semantic_audit.json'}`",
            f"- `final_object_semantic_audit.md`: `{scene_dir / 'final_object_semantic_audit.md'}`",
        ]
    )
    if audit_dir is not None:
        export_lines.extend(
            [
                f"- `local_memory_audit.jsonl`: `{audit_dir / 'local_memory_audit.jsonl'}`",
                f"- `local_memory_audit.log`: `{audit_dir / 'local_memory_audit.log'}`",
                f"- `local_memory_audit_summary.json`: `{audit_dir / 'local_memory_audit_summary.json'}`",
                f"- `local_memory_audit_vis/`: `{audit_dir / 'vis'}`",
            ]
        )
```

In the report config section, after frame count, add:

```python
            f"- `output_profile`: `{output_profile.name}`",
```

Replace the hard-coded `## Exports` list with:

```python
            *export_lines,
```

- [ ] **Step 10: Update run report JSON sidecar**

In the `sidecar = { ... }` dict in `write_run_report()`, add:

```python
        "output_profile": output_profile.name,
        "output_profile_artifacts": {
            "benchmark_audit_enabled": bool(output_profile.benchmark_audit_enabled),
            "write_debug_ply": bool(output_profile.write_debug_ply),
            "write_eval_mesh_ply": bool(output_profile.write_eval_mesh_ply),
            "write_preview_render": bool(output_profile.write_preview_render),
        },
```

Place these near the top next to metric fields.

- [ ] **Step 11: Update existing report test for default output profile**

In `test_write_run_report_describes_pool_and_tsdf_exports`, update the `write_run_report(...)` call to pass:

```python
        output_profile=build_output_profile(
            SimpleNamespace(
                fast_eval=False,
                full_debug=False,
                lightweight_benchmark=True,
                skip_debug_ply=False,
                skip_eval_ply=False,
                skip_preview_render=False,
            )
        ),
```

Add these assertions after reading `report_md` and `report_json`:

```python
    assert "`output_profile`: `custom`" in report_md
    assert "room0_instance_map_local_memory.ply" in report_md
    assert "room0_gtmesh_pred_semantic.ply" in report_md
    assert '"output_profile": "custom"' in report_json
```

- [ ] **Step 12: Add fast report listing test**

Append this test after `test_write_run_report_describes_pool_and_tsdf_exports`:

```python
def test_write_run_report_omits_fast_eval_debug_artifacts(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    eval_dir = tmp_path / "eval"
    scene_dir.mkdir()
    eval_dir.mkdir()
    obj = ObjectMap(
        object_id=1,
        state=ObjectState.ACTIVE,
        local_pcd=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        semantic_memory=SemanticMemory(label_hypotheses=[("chair", 0.9)]),
    )
    obj.debug["global_instance_substrate"] = {"stability_score": 0.8}
    state = SystemState(objects={1: obj})
    tsdf_records = np.zeros(0, dtype=build_tsdf_backbone_records(SystemState(objects={}, tsdf_volume=TSDFInstanceVolume(voxel_size=0.05))).dtype)
    pool_records = build_pool_semantic_records(state)
    dense_surface_records = np.zeros(0, dtype=pool_records.dtype)
    dense_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    evaluation = {
        "miou": 0.1,
        "macc": 0.2,
        "fmiou": 0.3,
        "fmacc": 0.4,
        "per_class": [{"class_name": "chair", "iou": 0.5, "acc": 0.6}],
    }
    frame_metrics = [
        {
            "raw_proposal_count": 1,
            "matched_patch_count": 1,
            "ambiguous_patch_count": 0,
            "ambiguous_matched_count": 0,
            "ambiguous_new_patch_count": 0,
            "local_memory_point_count_total": 1,
            "current_frame_visibility_gate": {
                "rejected_patch_count": 0,
                "depth_rejected_point_count": 0,
            },
        }
    ]
    args = SimpleNamespace(
        proposal_device="cuda",
        sam_version="2",
        sam_encoder="hiera_l",
        points_per_side=16,
        max_proposals=64,
    )
    pipeline = SimpleNamespace(proposal=SimpleNamespace(active_backend_name="precomputed"))
    output_profile = build_output_profile(
        SimpleNamespace(
            fast_eval=True,
            full_debug=False,
            lightweight_benchmark=False,
            skip_debug_ply=False,
            skip_eval_ply=False,
            skip_preview_render=False,
        )
    )

    write_run_report(
        scene_dir=scene_dir,
        eval_dir=eval_dir,
        experiment_name="test_fast_eval_report",
        dataset=SimpleNamespace(),
        args=args,
        frame_limit=1,
        pipeline=pipeline,
        state=state,
        tsdf_records=tsdf_records,
        pool_semantic_records=pool_records,
        dense_surface_records=dense_surface_records,
        dense_points=dense_points,
        labels=labels,
        evaluation=evaluation,
        frame_metrics=frame_metrics,
        audit_dir=None,
        output_profile=output_profile,
    )

    report_md = (scene_dir / "run_report.md").read_text(encoding="utf-8")
    report_json = (scene_dir / "run_report.json").read_text(encoding="utf-8")

    assert "`output_profile`: `fast_eval`" in report_md
    assert "room0_instance_map.ply" in report_md
    assert "room0_instance_map_dense_surface.ply" in report_md
    assert "room0_dense_geometry_instance_projected.ply" in report_md
    assert "room0_instance_map_local_memory.ply" not in report_md
    assert "room0_gtmesh_pred_semantic.ply" not in report_md
    assert "local_memory_audit.jsonl" not in report_md
    assert '"output_profile": "fast_eval"' in report_json
    assert '"write_debug_ply": false' in report_json
    assert '"write_eval_mesh_ply": false' in report_json
```

- [ ] **Step 13: Run artifact gating tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_dual_map.py::test_write_eval_artifacts_can_skip_mesh_plys \
  tests/test_dual_map.py::test_write_run_report_describes_pool_and_tsdf_exports \
  tests/test_dual_map.py::test_write_run_report_omits_fast_eval_debug_artifacts \
  -q
```

Expected: PASS.

- [ ] **Step 14: Commit**

```bash
git add run_room0_full_eval.py tests/test_dual_map.py
git commit -m "feat: gate room0 fast eval artifacts"
```

---

### Task 3: Fast Mode Smoke Validation

**Files:**
- No source changes expected.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python -m pytest \
  tests/test_dual_map.py \
  tests/test_pipeline.py::TestRoadmapRefactor::test_contested_patch_survives_parent_owned_surface_gate \
  tests/test_provisional_pool.py::test_contested_patch_survives_parent_owned_surface_gate \
  -q
```

Expected: PASS.

- [ ] **Step 2: Start a 0526-style fast smoke run**

Run from repository root:

```bash
tmux new-session -d -s room0_0526_fast_eval_20f /bin/bash -lc 'cd /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates && experiment=20260529_room0_0526_fast_eval_stride10_20f && log=outputs/tmp_validation/${experiment}.log && status=outputs/tmp_validation/${experiment}.status && { echo experiment=${experiment}; echo started_at=$(date --iso-8601=seconds); echo running > "${status}"; /home/ww/miniconda3/envs/oviovo/bin/python run_room0_full_eval.py --config-path configs/room0_surface_gate_4090.yaml --experiment-name "${experiment}" --num-frames 20 --frame-stride 10 --fast-eval --dataset-root /home/ww/vv/dataset/Replica/room0 --gt-labels /home/ww/vv/oviovo/data/input/replica_semantic_gt/room0.txt --gt-mesh-ply /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/mesh_semantic.ply --gt-info-json /home/ww/vv/dataset/Replica-Dataset/Replica_original/room_0/habitat/info_semantic.json --output-root /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates/outputs/tmp_validation --proposal-backend precomputed --proposal-cache-dir /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2 --proposal-cache-manifest /home/ww/vv/oviovo/outputs/frontend_proposals/room0_sam2_cache_full_highrecall_v2/manifest.json; rc=$?; echo finished_at=$(date --iso-8601=seconds); echo exit_status=${rc}; echo exit_status=${rc} > "${status}"; exit ${rc}; } > "${log}" 2>&1'
```

Expected immediately:

```bash
tmux list-sessions
```

shows `room0_0526_fast_eval_20f`.

- [ ] **Step 3: Monitor completion**

Run:

```bash
tail -n 80 outputs/tmp_validation/20260529_room0_0526_fast_eval_stride10_20f.log
cat outputs/tmp_validation/20260529_room0_0526_fast_eval_stride10_20f.status
```

Expected at completion:

```text
exit_status=0
```

- [ ] **Step 4: Verify fast artifact contract**

Run:

```bash
/home/ww/miniconda3/envs/oviovo/bin/python - <<'PY'
from pathlib import Path
import json

root = Path("outputs/tmp_validation/20260529_room0_0526_fast_eval_stride10_20f")
scene = root / "room0"
eval_dir = root / "replica"
required = [
    scene / "frame_metrics.jsonl",
    scene / "exports" / "room0_instance_map.ply",
    scene / "exports" / "room0_instance_map_dense_surface.ply",
    scene / "exports" / "room0_instance_map_tsdf_backbone.ply",
    scene / "exports" / "room0_dense_geometry_instance_projected.ply",
    scene / "final_object_semantic_audit.json",
    scene / "final_object_semantic_audit.md",
    scene / "run_report.json",
    scene / "run_report.md",
    eval_dir / "results.json",
    eval_dir / "classes_iou.json",
    eval_dir / "classes_acc.json",
    eval_dir / "statistics.txt",
]
for path in required:
    print("required", path, path.exists())
for path in [
    scene / "local_memory_audit",
    scene / "exports" / "room0_instance_map_local_memory.ply",
    scene / "vis" / "room0_dense_projection_preview.png",
    eval_dir / "room0_gtmesh_gt_semantic.ply",
    eval_dir / "room0_gtmesh_pred_semantic.ply",
    eval_dir / "room0_gtmesh_semantic_correctness.ply",
]:
    print("skipped", path, path.exists())
report = json.loads((scene / "run_report.json").read_text())
print("output_profile", report.get("output_profile"))
print("metrics", json.loads((eval_dir / "results.json").read_text()))
print("frame_metrics_lines", sum(1 for _ in (scene / "frame_metrics.jsonl").open()))
PY
```

Expected:

```text
All required paths print True.
All skipped paths print False.
output_profile fast_eval
frame_metrics_lines 20
```

- [ ] **Step 5: Compare artifact size against full 0526 output**

Run:

```bash
du -sh outputs/tmp_validation/20260526_room0_observation_first_fix52e1714_200f
du -sh outputs/tmp_validation/20260529_room0_0526_fast_eval_stride10_20f
find outputs/tmp_validation/20260529_room0_0526_fast_eval_stride10_20f -maxdepth 3 -type f -printf '%s %p\n' | sort -nr | head -n 20
```

Expected: Fast smoke has no local audit directory, no debug duplicate PLY, no GT mesh visualization PLYs, and much smaller output size for the same number of frames.

---

## Self-Review Checklist

- Spec coverage:
  - Fast mode preserves algorithm behavior and only gates output artifacts.
  - Local memory audit and overlay PNGs are disabled by `--fast-eval`.
  - Debug duplicate PLY and GT mesh visualization PLYs are disabled by `--fast-eval`.
  - Primary exports, final audit, JSON metrics, and reports remain.
  - `--full-debug` restores current full-output behavior.

- Placeholder scan:
  - No placeholder tasks or undefined function names remain.
  - Every code change includes concrete snippets and exact test commands.

- Type consistency:
  - `OutputProfile` fields match all usages in `main()` and `write_run_report()`.
  - `write_eval_artifacts(write_mesh_ply=...)` has a default preserving current behavior.
  - Existing tests can continue calling `write_run_report()` without passing an output profile because the function supplies a default.
