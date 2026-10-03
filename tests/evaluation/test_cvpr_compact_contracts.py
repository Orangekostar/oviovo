"""Fixed-cohort, camera visibility and unknown-aware output contracts."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from static_ovmap.module_validation.assets import sha256_file
from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.module_validation.evaluation import GeometryIdentity, PredictionPayload
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.recovery_wave2.recovery_registry import build_registry


SPEC = Path(__file__).resolve().parents[2] / "configs/static_ovmap/cvpr_compact_tables_v1.json"


def partial_scope_fixture(scene="office1"):
    from static_ovmap.cvpr_compact.partial_execution import partition_methods, semantic_block

    spec = json.loads(SPEC.read_text())
    selected = ["first", "second", "third"]
    projected = {"scene_id": scene, "GT_input": False, "failed_view_replacement": False,
        "g1": {"owner:86": selected[:1]}, "views": {"owner:86": selected},
        "requests": {rid: {"request_id": rid, "scene": scene, "frame_id": 1950 - i * 10,
                           "target_mask_sha256": str(i) * 64} for i, rid in enumerate(selected)}}
    projected["identity"] = canonical_digest(projected)
    failed = {"scene": scene, "status": "FAILED", "GT_input": False,
        "failed_view_replacement": False, "arms": ["G1", "G3", "U2"],
        "error": "RuntimeError: BLOCKED_ALL_SEMANTIC_REQUESTS_FAILED: selected requests exist",
        "requests": {rid: {**row, "status": "UNAVAILABLE_TECHNICAL_FAILURE",
                           "reason": "EMPTY_DENSE_MASK_SUPPORT"} for rid, row in projected["requests"].items()}}
    failed["identity"] = canonical_digest(failed)
    blocks = {arm + "_FC": semantic_block(failed, projected, arm) for arm in ("G1", "G3")}
    available_sources = ["G1_NATIVE", "ARCHIVED_U2_FC"]
    methods, blocked = partition_methods(spec, scene, available_sources, blocks)
    scope = {"scene": scene, "expected_method_ids": [row["id"] for row in spec["methods"]],
        "available_recovery_sources": available_sources, "blocked_sources": blocks, "blocked_methods": blocked}
    return spec, projected, failed, methods, scope


def test_partial_scope_keeps_fixed_recipes_and_proves_each_failed_selected_mask():
    from static_ovmap.cvpr_compact.partial_execution import partition_methods, semantic_block

    spec, projected, failed, methods, scope = partial_scope_fixture()
    assert [row["id"] for row in methods] == ["CT_A0_NATIVE", "CT_A1_E", "CT_A4_NATIVE_RECOVERY", "CT_H_U2"]
    assert set(scope["blocked_methods"]) == {"CT_A2_R", "CT_A3_ER", "CT_A5_FC_ONLY", "CT_G3"}
    assert scope["blocked_sources"]["G1_FC"]["selected_request_ids"] == ["first"]
    assert scope["blocked_sources"]["G3_FC"]["selected_request_ids"] == ["first", "second", "third"]
    assert len(spec["methods"]) == 8 and spec["methods"][3]["recovery"] == "G1_FC"
    with pytest.raises(ValueError, match="partition"):
        partition_methods(spec, "office1", ["G1_NATIVE"], scope["blocked_sources"])
    with pytest.raises(ValueError, match="partition"):
        partition_methods(spec, "office1", ["G1_NATIVE", "ARCHIVED_U2_FC", "G1_FC"], scope["blocked_sources"])
    changed = copy.deepcopy(failed)
    changed["requests"]["first"]["target_mask_sha256"] = "changed-mask"
    changed["identity"] = canonical_digest({k: v for k, v in changed.items() if k != "identity"})
    with pytest.raises(ValueError, match="selected"):
        semantic_block(changed, projected, "G1")


@pytest.mark.parametrize("change", ["no_selected", "missing_failure", "successful_request", "successful_receipt"])
def test_partial_semantic_block_rejects_unproven_failure_or_genuine_no_view(change):
    from static_ovmap.cvpr_compact.partial_execution import semantic_block

    _, projected, failed, _, _ = partial_scope_fixture()
    if change == "no_selected":
        projected["g1"]["owner:86"] = []
    elif change == "missing_failure":
        del failed["requests"]["first"]
    elif change == "successful_request":
        failed["requests"]["first"]["status"] = "COMPLETE"
    else:
        failed["status"] = "COMPLETE"
    for value in (projected, failed):
        value["identity"] = canonical_digest({k: v for k, v in value.items() if k != "identity"})
    with pytest.raises(ValueError):
        semantic_block(failed, projected, "G1")


def test_partial_pool_requires_all_eight_scenes_and_exact_available_method_scope():
    from static_ovmap.cvpr_compact.partial_execution import select_partial_pool_rows

    spec, _, _, methods, scope = partial_scope_fixture()
    receipts = {scene: {"status": "COMPLETE", "scene": scene,
        "rows": [{"status": "COMPLETE", "scene": scene, "method": row["id"],
                  "rank_mode": "OFFICIAL_CURRENT_CLASS"} for row in spec["methods"]]}
        for scene in reversed(spec["cohorts"]["replica8"])}
    receipts["office1"] = {**scope, "status": "PARTIAL_WITH_TECHNICAL_BLOCKS", "rows": [
        {"status": "COMPLETE", "scene": "office1", "method": row["id"],
         "rank_mode": "OFFICIAL_CURRENT_CLASS"} for row in methods]}
    rows = select_partial_pool_rows(spec, "replica8", "CT_A0_NATIVE", receipts)
    assert [row["scene"] for row in rows] == spec["cohorts"]["replica8"] and len(rows) == 8
    with pytest.raises(ValueError, match="BLOCKED_TECHNICAL"):
        select_partial_pool_rows(spec, "replica8", "CT_A3_ER", receipts)
    missing = {s: row for s, row in receipts.items() if s != "room2"}
    with pytest.raises(ValueError, match="complete fixed scene set"):
        select_partial_pool_rows(spec, "replica8", "CT_A0_NATIVE", missing)
    receipts["office1"]["rows"].pop()
    with pytest.raises(ValueError, match="scope"):
        select_partial_pool_rows(spec, "replica8", "CT_A0_NATIVE", receipts)


def test_partial_encode_continues_native_and_u2_without_retrying_failed_fc(monkeypatch, tmp_path):
    from static_ovmap.cvpr_compact import workflow

    _, _, _, _, scope = partial_scope_fixture()
    binding = {"output_root": str(tmp_path), "spec": str(SPEC), "fc": {"python": "fc-python"}}
    data = {"runtime": {"native_perception_python": "native-python"}}
    monkeypatch.setattr(workflow, "_context", lambda *args: (data, tmp_path / "context.json"))
    monkeypatch.setattr(workflow, "_exhausted_semantic_blocks", lambda *args: scope["blocked_sources"])
    calls = []
    def worker(binding, **kwargs):
        calls.append(kwargs)
        if kwargs["module"] == "recovery_run":
            assert "--arm" in kwargs["arguments"] and "U2" in kwargs["arguments"]
        return {"identity": kwargs["module"]}
    monkeypatch.setattr(workflow, "run_worker", worker)
    result = workflow._encode(binding, ["office1"])
    assert [row["module"] for row in calls] == ["native_region_worker", "recovery_run"]
    assert result["office1"]["Native"] == "native_region_worker"
    assert result["office1"]["U2"] == "recovery_run"
    assert workflow._phase_status(result) == "PARTIAL_WITH_TECHNICAL_BLOCKS"
    assert workflow._phase_status({"office0": {"status": "COMPLETE"}}) == "COMPLETE"


def test_compact_captured_frames_resolve_retired_membership_before_causal_registration(monkeypatch):
    from types import SimpleNamespace
    from static_ovmap.cvpr_compact.query_bridge import CapturedFrames
    from static_ovmap.module_validation.query_study import CapturedFrames as OriginalFrames
    from static_ovmap.module_validation.query_lineage import CurrentLineage
    from static_ovmap.module_validation.query_state import QueryPolicyState, NativeCombineState

    original = {"label_instances_scope": "all_known_labels",
        "aliases": [{"old_label": 54, "resolved_label": 51}],
        "label_instances": [{"segment_label": 5, "instance_label": 5},
            {"segment_label": 51, "instance_label": 0},
            {"segment_label": 54, "instance_label": 5}]}
    candidates = [SimpleNamespace(owner_id=5, frame_index=35, overlap_pixels=100,
        camera_pose=np.eye(4), request_id="request")]
    state, combine = QueryPolicyState("Q_GAIN", 1), NativeCombineState()
    lineage = CurrentLineage()
    lineage.advance(original, state, combine, np.ones((1, 1)))
    with pytest.raises(ValueError, match="no complete positive ancestry"):
        lineage.register_candidates(candidates)

    def load_current(self, index):
        self.current = {"snapshot": copy.deepcopy(original), "candidates": candidates}
        return self.current

    monkeypatch.setattr(OriginalFrames, "load", load_current)
    frames = object.__new__(CapturedFrames)
    frames._compact_aliases = {}
    normalized = frames.load(35)["snapshot"]
    assert original["label_instances"][-1]["instance_label"] == 5
    assert len(normalized["label_instances"]) == len(original["label_instances"])
    assert normalized["label_instances"][-1]["instance_label"] == 0
    state, combine = QueryPolicyState("Q_GAIN", 1), NativeCombineState()
    lineage = CurrentLineage()
    lineage.advance(normalized, state, combine, np.ones((1, 1)))
    lineage.register_candidates(candidates)
    assert lineage.requests["request"]["segments"] == frozenset({5})

    next_snapshot = copy.deepcopy(original)
    next_snapshot["aliases"] = []
    next_snapshot["label_instances"][0]["instance_label"] = 0
    frames.current = None
    original = next_snapshot
    normalized = frames.load(36)["snapshot"]
    assert normalized["label_instances"][-1]["instance_label"] == 0
    lineage.advance(normalized, state, combine, np.ones((1, 1)))
    with pytest.raises(ValueError, match="no complete positive ancestry"):
        lineage.register_candidates(candidates)


@pytest.mark.parametrize("aliases,rows,error", [
    ([{"old_label": 1, "resolved_label": 2}, {"old_label": 2, "resolved_label": 1}],
        [{"segment_label": 1, "instance_label": 7}, {"segment_label": 2, "instance_label": 7}], "cyclic"),
    ([{"old_label": 1, "resolved_label": 2}], [{"segment_label": 1, "instance_label": 7}], "missing"),
])
def test_compact_alias_bridge_rejects_unproven_current_membership(aliases, rows, error):
    from static_ovmap.cvpr_compact.query_bridge import normalize_alias_snapshot

    with pytest.raises(ValueError, match=error):
        normalize_alias_snapshot({"label_instances_scope": "all_known_labels",
            "aliases": aliases, "label_instances": rows}, {})


def test_compact_native_adapter_preserves_original_inference_and_replay():
    import ast
    import inspect
    from static_ovmap.backbone_wave1 import semantic_readout
    from static_ovmap.cvpr_compact.query_bridge import adapt_native_function, replay_captured
    from static_ovmap.module_validation.query_study import replay_captured as original_replay

    original = inspect.getsource(semantic_readout.run_native_query)
    worker, adapter = adapt_native_function(lambda *args: None, lambda *args: None)
    assert inspect.getsource(semantic_readout.run_native_query) == original
    assert adapter["original_AST"] == canonical_digest(ast.dump(ast.parse(original)))
    assert adapter["changed_AST_nodes"] == 2
    assert "static_ovmap.cvpr_compact.query_bridge" in worker.__code__.co_names
    assert inspect.signature(worker) == inspect.signature(semantic_readout.run_native_query)
    assert replay_captured is original_replay


def test_compact_alias_bridge_dispatch_preserves_failed_worker_log(tmp_path, monkeypatch):
    from static_ovmap.cvpr_compact import base_sources, query_bridge
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    captured = {}

    def execute(argv, cwd, log_path, **kwargs):
        captured.update(argv=argv, log_path=log_path, cwd=cwd)
        return {"status": "COMPLETE"}

    monkeypatch.setattr(base_sources, "execute_leaf", execute)
    binding = {"repository_root": str(tmp_path), "gpu": 2}
    job = {"output_root": str(tmp_path / "native_query")}
    result = query_bridge._run_worker(binding, "python", job, "native-query", ConsumptionIndex())
    assert result["status"] == "COMPLETE"
    assert captured["argv"][2] == "static_ovmap.cvpr_compact.query_bridge"
    assert captured["log_path"].name == "run.alias_bridge.log"


def plane(z=2., owner=7):
    xyz = np.array([[-3., -3., z], [3., -3., z], [3., 3., z], [-3., 3., z]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    return xyz, faces, np.full(4, owner, np.int64)


def camera():
    return np.array([[10., 0., 9.5], [0., 10., 9.5], [0., 0., 1.]]), np.eye(4)


def test_exact_cohort_counts_and_fixed_primary():
    from static_ovmap.cvpr_compact.protocol import load_spec, experiment_matrix

    spec = load_spec(SPEC)
    matrix = experiment_matrix(spec)
    assert len(matrix["anchors"]) == 26
    assert len(matrix["outputs"]) == 172
    assert len(matrix["pools"]) == 14
    assert len(matrix["timings"]) == 24
    assert spec["selection"]["primary_method"] == "CT_A3_ER"
    assert len({r["physical_family"] for r in matrix["anchors"] if r["cohort"] == "scannet_cf18"}) == 7
    assert [r["scene"] for r in matrix["anchors"] if r["cohort"] == "replica8"] == spec["cohorts"]["replica8"]


def test_final_temperatures_must_agree_across_all_eight_anchors():
    from static_ovmap.cvpr_compact.protocol import final_temperatures

    spec = json.loads(SPEC.read_text())
    vector = {"N": .01, "Q": .02, "F": .03}
    scenes = {s: {"temperatures": copy.deepcopy(vector)} for s in spec["cohorts"]["replica8"]}
    assert final_temperatures(spec, scenes) == vector
    scenes["room2"]["temperatures"]["F"] = .030000000000000002
    with pytest.raises(ValueError, match="final.*temperature"):
        final_temperatures(spec, scenes)


def test_camera_z_rays_are_not_range_normalized_and_batching_is_exact():
    from static_ovmap.cvpr_compact.projected_views import FullSceneProjector, camera_rays

    xyz, faces, raw = plane()
    K, pose = camera()
    rays = camera_rays(K, pose, (20, 20), 0, 400)
    np.testing.assert_array_equal(rays[:, :3], np.zeros((400, 3)))
    np.testing.assert_array_equal(rays[:, 5], np.ones(400))
    assert np.linalg.norm(rays[0, 3:]) > 1.5
    first = FullSceneProjector(xyz, faces, raw).project_frame(K, pose, np.full((20, 20), 2., np.float32), [7])
    second = FullSceneProjector(xyz, faces, raw, batch_pixels=73).project_frame(K, pose, np.full((20, 20), 2., np.float32), [7])
    assert first["masks"][7].sum() == 400
    np.testing.assert_array_equal(first["masks"][7], second["masks"][7])
    assert first["owners"][0, 0] == 7


def test_zero_and_mixed_owner_faces_still_occlude_far_candidates():
    from static_ovmap.cvpr_compact.projected_views import FullSceneProjector

    far, ff, fr = plane(2., 7)
    front, nf, nr = plane(1., 0)
    K, pose = camera()
    for front_owners in (nr, np.array([1, 2, 3, 4])):
        caster = FullSceneProjector(np.concatenate((front, far)), np.concatenate((nf, ff + 4)), np.concatenate((front_owners, fr)))
        result = caster.project_frame(K, pose, np.full((20, 20), 2., np.float32), [7])
        assert not result["masks"]
        assert result["counts"][7]["hit_owner_pixels"] == 0
        assert result["total_first_hits"] == 400


def test_invalid_depth_is_rejected_and_original_primitive_indices_survive():
    from static_ovmap.cvpr_compact.projected_views import FullSceneProjector

    xyz, faces, raw = plane()
    caster = FullSceneProjector(xyz, np.concatenate(([[0, 0, 0]], faces)), raw)
    assert caster.excluded_triangle_indices.tolist() == [0]
    assert caster.original_triangle_indices.tolist() == [1, 2]
    K, pose = camera()
    depth = np.full((20, 20), 2., np.float32)
    depth.flat[:4] = [0., np.nan, -1., 2.05]
    result = caster.project_frame(K, pose, depth, [7])
    assert result["masks"][7].sum() == 396
    assert result["counts"][7]["hit_owner_pixels"] == 400
    assert result["counts"][7]["depth_consistent_pixels"] == 396
    with pytest.raises(ValueError, match="batch"):
        FullSceneProjector(xyz, faces, raw, batch_pixels=65537)
    with pytest.raises(ValueError, match="indices"):
        FullSceneProjector(xyz, faces + 5, raw)


def test_projected_views_without_old_requests_have_g1_prefix_and_truthful_loader(tmp_path):
    from static_ovmap.cvpr_compact.projected_views import build_projected_views, load_projected_request

    xyz, faces, raw = plane()
    xyz = np.tile(xyz, (100, 1))
    raw = np.tile(raw, 100)
    _write_npz(tmp_path / "surface.npz", {"surface_xyz": xyz, "surface_faces": faces, "original_owner": raw})
    K, pose = camera()
    frames = []
    for fid in (30, 10, 20):
        frame_root = tmp_path / "frames" / str(fid)
        frame_root.mkdir(parents=True)
        image = np.zeros((20, 20, 3), np.uint8)
        image[:, :, 2] = 255
        assert cv2.imwrite(str(frame_root / "rgb.png"), image)
        _write_npz(frame_root / "depth.npz", {"depth_m": np.full((20, 20), 2., np.float32)})
        frames.append({"frame_id": fid, "scene_id": "fixture", "image_size_hw": [20, 20],
            "intrinsics": K.tolist(), "pose_c2w": pose.tolist(), "requests": [], "native_selected_request_ids": [],
            "rgb_path": f"frames/{fid}/rgb.png", "rgb_sha256": sha256_file(frame_root / "rgb.png"),
            "depth_path": f"frames/{fid}/depth.npz", "depth_sha256": sha256_file(frame_root / "depth.npz")})
    capture = {"scene_id": "fixture", "scheduled_frame_ids": [30, 10, 20], "completed_frame_ids": [30, 10, 20],
        "frames": frames, "surface": {"path": "surface.npz", "sha256": sha256_file(tmp_path / "surface.npz")}}
    capture["identity"] = canonical_digest(capture)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    geometry = GeometryIdentity(_array_digest(xyz), _array_digest(faces), "1" * 64, "projection", len(raw))
    native = PredictionPayload("N0", "N0", "fixture", geometry, np.zeros(len(raw), np.int64), np.zeros(len(raw), np.int64), (), {})
    native.lock()
    receipt = build_projected_views(path, native, tmp_path / "views")
    manifest = json.loads(Path(receipt["manifest"]).read_text())
    ids = manifest["views"]["owner:7"]
    assert [manifest["requests"][rid]["frame_id"] for rid in ids] == [10, 20, 30]
    assert manifest["g1"]["owner:7"] == ids[:1]
    request = manifest["requests"][ids[0]]
    assert request["schema"] == "ovimap-projected-region-request-v1"
    assert request["canonical_bbox"] == [0, 0, 20, 20]
    assert request["legacy_native_bbox"] == [0, 0, 19, 19]
    assert "native_selected_request_ids" not in request
    value = load_projected_request(Path(receipt["manifest"]), ids[0])
    assert value["target"].dtype == np.bool_ and value["target"].sum() == 400
    np.testing.assert_array_equal(value["target"], value["union"])
    np.testing.assert_array_equal(value["image"][0, 0], [255, 0, 0])
    assert build_projected_views(path, native, tmp_path / "views")["identity"] == receipt["identity"]


def test_a5_keeps_unknown_incumbent_support_and_appends_only_residual_rows():
    from static_ovmap.cvpr_compact.outputs import construct_output, fc_only_labels

    xyz = np.column_stack((np.arange(8), np.zeros((8, 2)))).astype(np.float32)
    raw = np.array([7, 7, 9, 9, 9, 9, 12, 12])
    painted = np.array([7, 7, 0, 0, 0, 0, 0, 0])
    geometry = GeometryIdentity(_array_digest(xyz), "1" * 64, "2" * 64, "projection", 8)
    baseline = PredictionPayload("N0", "N0", "fixture", geometry, painted, np.array([1, 1, 0, 0, 0, 0, 0, 0]), ((7, 1.),), {})
    baseline.lock()
    registry = build_registry(xyz, raw, painted, minimum_rows=2)
    F = {"objects": {"7": {"available": False, "scores": None}}}
    incumbents = fc_only_labels(F, [1, 2], {7: 1})
    assert incumbents == {7: 0}
    nearest = np.tile(np.arange(8), 100)
    output = construct_output(baseline, raw, registry, {9: 2}, [1, 2], nearest,
        np.ones(800, bool), "CT_A5_FC_ONLY", incumbent_labels=incumbents)
    np.testing.assert_array_equal(output.owner_ids, [7, 7, 9, 9, 9, 9, 0, 0])
    np.testing.assert_array_equal(output.semantic_labels, [0, 0, 2, 2, 2, 2, 0, 0])
    assert output.metadata["owner_semantic_decisions"] == {"7": 0, "9": 2}
    assert dict(output.instance_ranks) == {7: 0., 9: 1.}
    from static_ovmap.m2_reviewer_study.evaluation import official_view
    assert set(official_view(output.owner_ids[nearest], {7: 0, 9: 2}, 100)) == {9}
    with pytest.raises(ValueError, match="positive"):
        construct_output(baseline, raw, registry, {9: 0}, [1, 2], nearest,
            np.ones(800, bool), "CT_A5_FC_ONLY", incumbent_labels=incumbents)
    with pytest.raises(ValueError, match="registry"):
        construct_output(baseline, raw, registry, {123: 1}, [1, 2], nearest,
            np.ones(800, bool), "CT_A3_ER")


def test_released_overlap_noise_is_tolerated_but_vector_is_preserved():
    from static_ovmap.cvpr_compact.protocol import validate_overlaps

    original = np.r_[np.arange(.5, .95, .05), .25]
    assert validate_overlaps(original) is original
    with pytest.raises(ValueError, match="overlap"):
        validate_overlaps(np.r_[np.arange(.5, 1., .05), .25])


def test_fixed_download_plan_never_selects_development_or_random_families():
    from static_ovmap.cvpr_compact.benchmark_inputs import fixed_download_plan
    from static_ovmap.module_validation.scannet_download import DownloaderLayout, REQUIRED_SUFFIXES

    spec = json.loads(SPEC.read_text())
    layout = DownloaderLayout("https://kaldir.vc.cit.tum.de/scannet/", ("v2/scans", "v1/scans"),
        ("v2/tasks", "v1/tasks"), ("scannetv2-labels.combined.tsv", "scannet-labels.combined.tsv"),
        "/fixture/downloader.py", "0" * 64)
    plan = fixed_download_plan(spec, layout, Path("/fixture/data"))
    assert [r["scene_id"] for r in plan["captures"]] == spec["cohorts"]["scannet_cf18"]
    for row in plan["captures"]:
        assert set(row["files"]) == set(REQUIRED_SUFFIXES)
        for suffix, item in row["files"].items():
            assert ("/v1/scans/" if suffix == ".sens" else "/v2/scans/") in item["url"]
            assert Path(item["path"]).name == row["scene_id"] + suffix
    assert not set(spec["development_scenes"]) & {r["scene_id"] for r in plan["captures"]}


def test_repeatable_readonly_path_maps_reject_conflicts(tmp_path):
    from static_ovmap.cvpr_compact.binding import parse_path_maps

    path = tmp_path / "maps.json"
    path.write_text(json.dumps({"/old/two": "/new/two"}))
    assert parse_path_maps(["/old/one=/new/one", str(path)]) == {
        "/old/one": "/new/one", "/old/two": "/new/two"}
    with pytest.raises(ValueError, match="conflict"):
        parse_path_maps(["/old/one=/new/one", "/old/one=/different"])


def test_region_plans_keep_fixed_g1_prefix_and_do_not_replace_failed_views():
    from static_ovmap.cvpr_compact.region_worker import projected_plan, classify_regions

    registry = {"candidates": [{"raw_owner": 7}, {"raw_owner": 9}], "identity": "registry"}
    requests = {str(i): {"frame_id": i, "visible_pixels": 400 - i * 100,
        "target_mask_sha256": str(i), "raw_owner": 7, "selection_rank": i} for i in range(3)}
    manifest = {"requests": requests, "views": {"owner:7": ["0", "1", "2"], "owner:9": []},
        "g1": {"owner:7": ["0"], "owner:9": []}, "registry_identity": "registry", "identity": "views"}
    plan = projected_plan(registry, manifest, ("G1", "G3"))
    assert plan["G1"]["7"] == ["0"] and plan["G3"]["7"] == ["0", "1", "2"]
    features = {"1": np.array([1., 0.], np.float32), "2": np.array([0., 1.], np.float32)}
    sources = classify_regions(registry, plan, requests, features, np.eye(2), [1, 2])
    assert not sources["G1"]["7"]["available"]
    assert not sources["G3"]["9"]["available"]
    assert sources["G3"]["7"]["used_request_ids"] == ["1", "2"]
    assert sources["G3"]["7"]["failed_request_ids"] == ["0"]
    np.testing.assert_allclose(sources["G3"]["7"]["scores"], [3 / np.sqrt(13), 2 / np.sqrt(13)])
    assert sources["G3"]["7"]["label"] == 1
    bad = copy.deepcopy(manifest)
    bad["g1"]["owner:7"] = ["1"]
    with pytest.raises(ValueError, match="prefix"):
        projected_plan(registry, bad, ("G1",))
    bad = copy.deepcopy(manifest)
    bad["requests"]["2"]["frame_id"] = 1
    with pytest.raises(ValueError, match="distinct"):
        projected_plan(registry, bad, ("G3",))


def test_zero_region_aggregate_is_unavailable_and_nonunit_evidence_is_rejected():
    from static_ovmap.cvpr_compact.region_worker import classify_regions, region_feature_key

    registry = {"candidates": [{"raw_owner": 7}]}
    plan = {"G3": {"7": ["a", "b"]}}
    requests = {rid: {"visible_pixels": 100} for rid in ("a", "b")}
    features = {"a": np.array([1., 0.]), "b": np.array([-1., 0.])}
    row = classify_regions(registry, plan, requests, features, np.eye(2), [1, 2])["G3"]["7"]
    assert not row["available"] and row["reason"] == "ZERO_REGION_AGGREGATE"
    assert row["label"] is None and row["scores"] is None
    with pytest.raises(ValueError, match="unit"):
        classify_regions(registry, plan, requests, {"a": np.array([2., 0.])}, np.eye(2), [1, 2])
    mask = np.ones((20, 20), bool)
    assert region_feature_key("tensor-a", mask) != region_feature_key("tensor-b", mask)
    other = mask.copy(); other[0, 0] = False
    assert region_feature_key("tensor-a", mask) != region_feature_key("tensor-a", other)


def test_fc_execution_shares_one_frame_but_separates_mask_features_and_cold_reads(tmp_path, monkeypatch):
    import torch
    from static_ovmap.a7_evidence_upgrade import region_adapter
    from static_ovmap.cvpr_compact.region_worker import FCSession
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex
    from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache

    image = np.zeros((4, 4, 3), np.uint8)
    image[:, :2, 0] = 100
    image[:, 2:, 1] = 100
    left = np.zeros((4, 4), bool); left[:, :2] = True
    right = ~left
    calls = {"encode": 0, "pool": 0}
    def image_tensor(rgb, device):
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float(), rgb.shape[:2]
    def signed_mask(mask, size, padded, dense_size, device):
        signed = torch.from_numpy(mask.astype(np.float32) * 2 - 1)[None, None]
        return signed, signed > 0, mask
    def extract(model, tensor):
        calls["encode"] += 1
        return {"clip_vis_dense": tensor[:, :2]}
    def region_vector(model, ops, dense, signed):
        calls["pool"] += 1
        selected = signed[0, 0] > 0
        vector = dense[0, :, selected].mean(dim=1)
        return torch.nn.functional.normalize(vector, dim=0), int(selected.sum())
    monkeypatch.setattr(region_adapter, "image_tensor", image_tensor)
    monkeypatch.setattr(region_adapter, "signed_mask", signed_mask)
    monkeypatch.setattr(region_adapter, "region_vector", region_vector)
    session = FCSession.__new__(FCSession)
    session.model_key, session.device, session.cache = "model", "cpu", None
    session.text = np.eye(2, dtype=np.float32)
    session.index, session.model = ConsumptionIndex(), object()
    session.operators = {"extract_features_convnext": extract}
    session.load_model = lambda: session.model
    requests = {rid: {"image_sha256": "same-image", "frame_id": 1} for rid in ("left", "right", "left-alias")}
    masks = {"left": left, "right": right, "left-alias": left}
    def loader(rid):
        return {"image": image, "target": masks[rid]}
    loaders = {rid: loader for rid in requests}
    features, stats = session.encode(requests, loaders)
    assert calls == {"encode": 1, "pool": 2}
    assert stats["physical_image_encodings"] == 1 and stats["physical_region_poolings"] == 2
    assert not stats["persistent_feature_cache_enabled"]
    np.testing.assert_array_equal(features["left"], [1., 0.])
    np.testing.assert_array_equal(features["right"], [0., 1.])
    assert stats["requests"]["left-alias"]["within_run_region_alias"] == "left"
    cache = ContentCache([], tmp_path / "features", "model", index=session.index)
    session.cache = cache
    warm_features, first = session.encode(requests, loaders)
    assert first["physical_image_encodings"] == 1 and first["physical_region_poolings"] == 2
    _, second = session.encode(requests, loaders)
    assert second["physical_image_encodings"] == second["physical_region_poolings"] == 0
    assert second["region_cache_hits"] == 3
    def forbidden(*args):
        raise AssertionError("cold replay attempted a persistent feature read")
    monkeypatch.setattr(cache, "lookup", forbidden)
    session.cache = None
    cold_features, cold = session.encode(requests, loaders)
    assert cold["physical_image_encodings"] == 1 and cold["physical_region_poolings"] == 2
    for rid in requests:
        np.testing.assert_array_equal(warm_features[rid], cold_features[rid])
    def broken_encoder(model, tensor):
        raise RuntimeError("synthetic fatal encoder failure after payment")
    session.operators["extract_features_convnext"] = broken_encoder
    with pytest.raises(RuntimeError, match="after payment"):
        session.encode(requests, loaders)
    assert session.last_stats["physical_image_encodings"] == 1
    assert all(row["status"] == "UNAVAILABLE_TECHNICAL_FAILURE"
               for row in session.last_stats["requests"].values())


def test_native_recovery_keeps_actual_six_crop_bbox_and_fp32_readout():
    from static_ovmap.cvpr_compact.native_region_worker import native_evidence_key, native_sources
    from static_ovmap.recovery_wave2.recovery_sources import single_native_classifier

    registry = {"candidates": [{"raw_owner": 7}, {"raw_owner": 9}]}
    plan = {"7": ["first"], "9": []}
    text = np.eye(2, dtype=np.float32)
    canonical = np.array([[1., 0.], [0., 1.]], np.float32)
    feature = np.array([.3, .1], np.float32)
    result = native_sources(registry, plan, {"first": feature}, text, canonical, [1, 2])
    expected = single_native_classifier(feature, text, canonical, [1, 2])
    assert result["7"]["scores"] == expected["scores"]
    assert result["7"]["precision"] == "float32" and result["7"]["label"] == 1
    assert not result["9"]["available"] and result["9"]["label"] is None
    request = {"image_sha256": "image", "target_mask_sha256": "mask", "legacy_native_bbox": [0, 0, 19, 19]}
    first = native_evidence_key(request, "native-model", "native-operators")
    other = {**request, "legacy_native_bbox": [0, 0, 20, 20]}
    assert first != native_evidence_key(other, "native-model", "native-operators")
    assert first != native_evidence_key(request, "other-model", "native-operators")
    with pytest.raises(ValueError, match="single"):
        native_sources(registry, {"7": ["first", "other"], "9": []}, {}, text, canonical, [1, 2])


def test_eight_outputs_keep_fixed_support_and_common_fc_recovery():
    from static_ovmap.cvpr_compact.outputs import build_method_outputs
    from static_ovmap.cvpr_compact.costs import native_usage

    xyz = np.column_stack((np.arange(8), np.zeros((8, 2)))).astype(np.float32)
    raw = np.array([7, 7, 9, 9, 9, 9, 12, 12])
    painted = np.array([7, 7, 0, 0, 0, 0, 0, 0])
    geometry = GeometryIdentity(_array_digest(xyz), "1" * 64, "2" * 64, "projection", 8)
    native = PredictionPayload("N0", "N0", "fixture", geometry, painted,
        np.array([1, 1, 0, 0, 0, 0, 0, 0]), ((7, 1.),), {})
    native.lock()
    registry = build_registry(xyz, raw, painted, minimum_rows=2)
    def source(name, owners, labels):
        value = {"source": name, "valid_ids": [1, 2], "objects": {str(owner): {
            "available": owner in labels, "scores": ([1., 0.] if labels.get(owner) == 1 else [0., 1.]) if owner in labels else None,
            "label": labels.get(owner), "attempted_request_ids": [], "used_request_ids": []} for owner in owners}}
        value["identity"] = canonical_digest(value)
        return value
    sources = {"N": source("N", [7], {7: 1}), "Q": source("Q", [7], {}), "F": source("F", [7], {7: 2})}
    recovery = {"G1_FC": source("G1", [9, 12], {9: 2}), "G1_NATIVE": source("G1_NATIVE", [9, 12], {9: 1}),
                "G3_FC": source("G3", [9, 12], {9: 1}), "ARCHIVED_U2_FC": source("U2", [9, 12], {12: 2})}
    spec = json.loads(SPEC.read_text())
    result = build_method_outputs(native, raw, registry, sources, {"N": .1, "Q": .1, "F": .1},
        [1, 2], np.tile(np.arange(8), 100), np.ones(800, bool), recovery, spec["methods"])
    assert set(result) == {row["id"] for row in spec["methods"]}
    for method in ("CT_A0_NATIVE", "CT_A1_E"):
        np.testing.assert_array_equal(result[method].owner_ids, painted)
    for method in ("CT_A2_R", "CT_A3_ER", "CT_A5_FC_ONLY"):
        assert result[method].metadata["recovered_labels"] == {"9": 2}
        np.testing.assert_array_equal(result[method].owner_ids, [7, 7, 9, 9, 9, 9, 0, 0])
    assert result["CT_A4_NATIVE_RECOVERY"].metadata["recovered_labels"] == {"9": 1}
    assert result["CT_H_U2"].metadata["recovered_labels"] == {"12": 2}
    assert result["CT_G3"].metadata["recovered_labels"] == {"9": 1}
    assert result["CT_A5_FC_ONLY"].semantic_labels[:2].tolist() == [2, 2]
    assert dict(result["CT_A5_FC_ONLY"].instance_ranks)[7] == .5
    np.testing.assert_array_equal(result["CT_A0_NATIVE"].semantic_labels, native.semantic_labels)
    empty = native_usage({}, [], "empty")
    inventory = {"status": "COMPLETE", "scene": "fixture", "support": native_usage({"native-tensor": 6}, [], "N_SUPPORT"),
                 "query": empty, "F": empty, "common_map_receipt": {"sha256": "map"}}
    inventory["identity"] = canonical_digest(inventory)
    costed = build_method_outputs(native, raw, registry, sources, {"N": .1, "Q": .1, "F": .1},
        [1, 2], np.tile(np.arange(8), 100), np.ones(800, bool), recovery, spec["methods"],
        cost_inventory=inventory, recovery_costs={name: empty for name in recovery})
    for name, payload in costed.items():
        np.testing.assert_array_equal(payload.owner_ids, result[name].owner_ids)
        np.testing.assert_array_equal(payload.semantic_labels, result[name].semantic_labels)
        assert payload.logical_cost["native_required_crop_inputs"] == 6
        assert payload.metadata["dependency_costs"]["native_support_prerequisite_included"]


def test_main_execution_requires_committed_implementation_freeze(tmp_path):
    from static_ovmap.cvpr_compact.runtime import require_frozen_execution

    spec = json.loads(SPEC.read_text())
    binding = {"spec": str(SPEC), "cohorts": spec["cohorts"], "output_root": str(tmp_path),
        "repository_root": str(tmp_path), "scenes": {"scene0056_00": {
            "availability": "REUSABLE_VERIFIED_BB00_NATIVE_ANCHOR"}}}
    binding["identity"] = canonical_digest(binding)
    assert require_frozen_execution(binding, "scene0056_00") == "DEVELOPMENT_ONLY_EXISTING_ANCHOR"
    with pytest.raises(RuntimeError, match="freeze"):
        require_frozen_execution(binding, "room0")
    with pytest.raises(ValueError, match="fixed"):
        require_frozen_execution(binding, "scene0534_00")


def test_missing_anchor_uses_exact_native_options_and_original_frontend(tmp_path):
    from static_ovmap.cvpr_compact.anchor import anchor_plan
    from static_ovmap.module_validation.assets import native_schedule
    from static_ovmap.backbone_wave1.binding import option

    command = ["/runtime/map/python", "/upstream/scripts/panoptic_mapping_.py",
        "--dataset", "scannet_nyu", "--task", "Nyu40", "--scene_num", "scene0056_00",
        "--data_folder", "/old/exported", "--result_folder", "/old/mapper", "--start", "0",
        "--end", "1792", "--step", "9", "--num_threads", "8", "--data_association", "2",
        "--inst_association", "4", "--seg_graph_confidence", "3", "--use_inst_label_connect", "1",
        "--connection_ratio_th", "0.2", "--use_temp_panoptics", "--temp_panoptics_folder", "/old/frontend",
        "--temp_geometrics_folder", "/old/geometrics", "--save_temp_results", "--intermediate_seg_folder", "/old/segments",
        "--perception_python", "/runtime/map/python", "--perception_worker", "/old/worker.py",
        "--use_temp_geometrics", "--skip_feature_extraction"]
    runtime = {"cropformer_root": "/cropformer", "cropformer_config": "/cropformer/config.yaml",
        "cropformer_weights": "/models/cropformer.pth", "frontend_python": "/runtime/frontend/bin/python",
        "baseline_build": "/baseline", "native_model": "/models/native", "hf_modules_cache": "/models/hf_modules"}
    template = {"actual_capture_command": {"command": command, "environment": {"OVIMAP_NATIVE_UPSTREAM": "/upstream"}},
        "original_map_options": {"id": "BB00_NATIVE", "frontend": "cropformer", "association": "native", "depth_fusion": "native"},
        "native_extension": {"path": "/bb00/lib/consistent_gsm.so"}, "runtime": runtime}
    binding = {"repository_root": str(tmp_path / "repo"), "output_root": str(tmp_path / "attempt"), "gpu": "2",
        "cohorts": {"scannet_cf18": ["scene0011_00"]}, "scenes": {"scene0056_00": template}}
    preparation = {"scene": "scene0011_00", "status": "COMPLETE", "export_root": str(tmp_path / "exported/scene0011_00"),
        "schedule": native_schedule(2374), "invalid_pose_frame_ids": []}
    plan = anchor_plan(binding, "scene0011_00", preparation)
    mapping = plan["mapping_command"]
    assert option(mapping, "--scene_num") == "scene0011_00"
    assert option(mapping, "--data_folder") == str(tmp_path / "exported")
    assert option(mapping, "--inst_association") == "4" and option(mapping, "--data_association") == "2"
    assert option(mapping, "--step") == "11" and option(mapping, "--end") == "2190"
    assert "--save_temp_geometrics" in mapping and "--use_temp_geometrics" not in mapping
    assert mapping.count("--skip_feature_extraction") == 1
    assert plan["mapping_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert plan["mapping_environment"]["PYTHONPATH"].split(":")[0] == "/bb00/lib"
    assert plan["recipe"]["order_diagnostics"] is False
    front = plan["frontend_command"]
    assert front[0] == runtime["frontend_python"] and front[1] == "/cropformer/demo_cropformer/demo_from_dirs.py"
    assert front[front.index("--out-type") + 1] == "0"
    assert front[front.index("--confidence-threshold") + 1] == "0.5"
    assert len(front[front.index("--input") + 1:front.index("--output")]) == 200
    assert template["actual_capture_command"]["command"] == command
    with pytest.raises(ValueError, match="fixed"):
        anchor_plan(binding, "scene0534_00", preparation)


def test_existing_e_uses_cosine_n_and_exact_final_reconciled_query_rows():
    from static_ovmap.cvpr_compact.base_sources import existing_sources
    from static_ovmap.module_validation.scannet_study import native_readout

    text = np.eye(2, dtype=np.float32)
    canonical = np.array([[1., 0.], [0., 1.]], np.float32)
    saved = {7: {"feat": np.array([[.6, .2], [.3, .1]], np.float32), "frame_id": [0, 11],
                 "vis_area": np.array([100, 200])}}
    query = {"valid_ids": np.array([1, 2]), "owner_ids": np.array([7, 9]),
        "available": np.array([True, False]), "scores": np.array([[.12, .84], [0., 0.]])}
    result = existing_sources({7: 1, 9: 2}, saved, query, text, canonical, [1, 2], "anchor", "native-model", "query-receipt")
    np.testing.assert_allclose(result["N"]["objects"]["7"]["scores"], [3 / np.sqrt(10), 1 / np.sqrt(10)], atol=1e-7)
    assert result["Q"]["objects"]["7"]["scores"] == [.12, .84]
    assert not result["N"]["objects"]["9"]["available"]
    assert not result["Q"]["objects"]["9"]["available"]
    assert native_readout(saved, text, canonical, (1, 2))[7]["class_id"] == 1
    assert result["N"]["objects"]["7"]["score_kind"] == "ORIGINAL_NATIVE_COSINE_FOR_D2"
    with pytest.raises(ValueError, match="vocabulary"):
        existing_sources({7: 1}, saved, {**query, "valid_ids": np.array([2, 1])},
                         text, canonical, [1, 2], "anchor", "native-model", "query-receipt")


def test_existing_f_keeps_original_selection_and_cap_exclusions_unavailable():
    from static_ovmap.cvpr_compact.base_sources import static_fc_plan
    from static_ovmap.module_validation.contracts import canonical_digest

    manifest = {"selected_targets": ["owner:7"], "excluded_targets": ["owner:9"],
        "views": {"owner:7": ["later", "earlier"]},
        "requests": {"later": {"frame_id": 11, "visible_target_pixels": 200},
                     "earlier": {"frame_id": 0, "visible_target_pixels": 100}},
        "lineage_proofs": {rid: {"target_id": "owner:7"} for rid in ("later", "earlier")}}
    manifest["identity"] = canonical_digest(manifest)
    plan, requests = static_fc_plan(manifest, {7: 1, 9: 2})
    assert plan == {"F": {"7": ["later", "earlier"], "9": []}}
    assert requests["later"]["visible_pixels"] == 200
    assert requests["earlier"]["frame_id"] == 0
    excessive = {k: v for k, v in manifest.items() if k != "identity"}
    excessive["views"] = {"owner:7": ["later", "earlier", "third", "fourth"]}
    excessive["identity"] = canonical_digest(excessive)
    with pytest.raises(ValueError, match="three"):
        static_fc_plan(excessive, {7: 1, 9: 2})
    with pytest.raises(ValueError, match="registry"):
        static_fc_plan(manifest, {7: 1, 9: 2, 10: 1})


def test_all_fresh_base_source_entries_gate_before_models_and_outputs(tmp_path):
    from static_ovmap.cvpr_compact.base_sources import run_base_sources, run_existing_fc, run_native_query
    from static_ovmap.cvpr_compact.query_bridge import run_base_sources as bridged_sources, run_native_query as bridged_native

    spec = json.loads(SPEC.read_text())
    binding = {"spec": str(SPEC), "cohorts": spec["cohorts"], "output_root": str(tmp_path),
        "repository_root": str(tmp_path), "scenes": {}}
    binding["identity"] = canonical_digest(binding)
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    job = {"binding": str(path), "scene": "scene0011_00"}
    for call in (lambda: run_base_sources(binding, job["scene"]),
                 lambda: run_native_query(job), lambda: run_existing_fc(job),
                 lambda: bridged_sources(binding, job["scene"]), lambda: bridged_native(job)):
        with pytest.raises(RuntimeError, match="freeze"):
            call()
    assert list(tmp_path.iterdir()) == [path]


def test_score_alias_requires_full_arrays_ranks_and_actual_scorer_context():
    from static_ovmap.cvpr_compact.evaluation import equivalent_scoring_input

    geometry = GeometryIdentity("1" * 64, "2" * 64, "3" * 64, "projection", 2)
    def prediction(labels=(1, 1), rank=1.):
        value = PredictionPayload("fixture", "COMBO", "fixture", geometry, np.array([7, 7]),
            np.array(labels), ((7, rank),), {})
        value.lock()
        return value
    context = {"gt": {"path": "/first/gt.npy", "sha256": "a" * 64, "bytes": 100},
        "native_prediction_key": "first", "valid_ids": [1, 2], "runtime_overlaps": [.5, .25]}
    relocated = {**context, "gt": {**context["gt"], "path": "/second/gt.npy"},
                 "native_prediction_key": "second"}
    view = {"7": {"label": 1, "rank": "1.000000", "area": 100}}
    assert equivalent_scoring_input(prediction(), prediction(), context, relocated, view, view)
    assert not equivalent_scoring_input(prediction(), prediction((2, 2)), context, relocated, view, view)
    assert not equivalent_scoring_input(prediction(), prediction(rank=.5), context, relocated, view, view)
    changed_gt = {**relocated, "gt": {**relocated["gt"], "sha256": "b" * 64}}
    assert not equivalent_scoring_input(prediction(), prediction(), context, changed_gt, view, view)
    changed_rank = {"7": {**view["7"], "rank": ".500000"}}
    assert not equivalent_scoring_input(prediction(), prediction(), context, relocated, view, changed_rank)


def test_pool_selection_requires_all_fixed_scenes_once_in_protocol_order():
    from static_ovmap.cvpr_compact.evaluation import select_pool_rows

    spec = json.loads(SPEC.read_text())
    scenes = spec["cohorts"]["scannet_cf18"]
    receipts = {scene: {"status": "COMPLETE", "scene": scene, "rows": [{"status": "COMPLETE",
        "scene": scene, "method": "CT_A3_ER", "rank_mode": "OFFICIAL_CURRENT_CLASS"}]} for scene in reversed(scenes)}
    rows = select_pool_rows(spec, "scannet_cf18", "CT_A3_ER", receipts)
    assert [row["scene"] for row in rows] == scenes
    with pytest.raises(ValueError, match="complete"):
        select_pool_rows(spec, "scannet_cf18", "CT_A3_ER", {s: r for s, r in receipts.items() if s != scenes[0]})
    receipts[scenes[0]]["rows"] *= 2
    with pytest.raises(ValueError, match="once"):
        select_pool_rows(spec, "scannet_cf18", "CT_A3_ER", receipts)
    with pytest.raises(ValueError, match="cohort"):
        select_pool_rows(spec, "scannet_cf18", "CT_G3", receipts)


def test_released_scoring_keeps_recovered_masks_and_unknown_semantic_errors(tmp_path, monkeypatch):
    import os
    from static_ovmap.cvpr_compact.evaluation import evaluate_scene
    from static_ovmap.cvpr_compact.outputs import build_method_outputs
    from static_ovmap.module_validation.boundary_jobs import file_identity
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.module_validation.scannet_study import save_prediction
    from static_ovmap.released_loader import load_released_module

    upstream = Path(os.environ.get("OVIMAP_NATIVE_TEST_UPSTREAM", "/home/ww/crove/ovimap-backbone-wave1-upstream"))
    if not (upstream / "scripts/eval_utils.py").is_file():
        pytest.skip("the pinned original scorer must be supplied via OVIMAP_NATIVE_TEST_UPSTREAM")
    constants = load_released_module(upstream / "scripts/utils/semantic_const.py")
    names = constants["REPLICA_51"]
    first, second = [(["background", *names]).index(name) for name in constants["CLASS_LABELS_REPLICA"][:2]]
    ids = list(range(1, 52))
    xyz = np.column_stack((np.arange(200), np.zeros((200, 2)))).astype(np.float32)
    raw = np.repeat([7, 9], 100)
    painted = np.repeat([7, 0], 100)
    geometry = GeometryIdentity(_array_digest(xyz), "2" * 64, "3" * 64, "synthetic-projection", 200)
    native = PredictionPayload("N0", "N0", "scene0056_00", geometry, painted, np.repeat([first, 0], 100), ((7, 1.),), {})
    native.lock()
    registry = build_registry(xyz, raw, painted)
    def source(name, owners, labels):
        objects = {}
        for owner in owners:
            score = np.zeros(51)
            if owner in labels:
                score[labels[owner] - 1] = 1.
            objects[str(owner)] = {"available": owner in labels, "scores": score.tolist() if owner in labels else None,
                "label": labels.get(owner), "attempted_request_ids": [], "used_request_ids": []}
        value = {"source": name, "valid_ids": ids, "objects": objects}
        value["identity"] = canonical_digest(value)
        return value
    sources = {"N": source("N", [7], {7: first}), "Q": source("Q", [7], {}), "F": source("F", [7], {})}
    recovery = {name: source(name, [9], {9: second}) for name in ("G1_FC", "G1_NATIVE", "G3_FC", "ARCHIVED_U2_FC")}
    spec = json.loads(SPEC.read_text())
    outputs = build_method_outputs(native, raw, registry, sources, {"N": .1, "Q": .1, "F": .1},
        ids, np.arange(200), np.ones(200, bool), recovery, spec["methods"])
    projection_dir = tmp_path / "projection"
    projection_dir.mkdir()
    np.savez_compressed(projection_dir / "projection.npz", nearest=np.arange(200), matched=np.ones(200, bool))
    projection_path = projection_dir / "manifest.json"
    atomic_write_json(projection_path, {"identity": geometry.projection_identity,
        "sha256": sha256_file(projection_dir / "projection.npz"), "source_rows": 200, "target_rows": 200})
    gt_semantic, gt_instance = np.repeat([first, second], 100), np.repeat([1, 2], 100)
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    np.savez_compressed(annotation_dir / "annotations.npz", xyz=xyz, gt_semantic=gt_semantic,
                        gt_instance=gt_instance, valid_ids=np.array(ids))
    np.save(annotation_dir / "gt_sem_inst_id.npy", gt_semantic * 1000 + gt_instance)
    annotation_path = annotation_dir / "receipt.json"
    atomic_write_json(annotation_path, {"status": "COMPLETE", "input_identity": "synthetic-annotations", "inputs": [],
        "arrays_path": str(annotation_dir / "annotations.npz"), "gt_instance_path": str(annotation_dir / "gt_sem_inst_id.npy"),
        "outputs": [file_identity(annotation_dir / name) for name in ("annotations.npz", "gt_sem_inst_id.npy")]})
    atomic_write_json(tmp_path / "source_manifest.json", {"entries": [file_identity(path) for path in (projection_path, annotation_path)]})
    config_path = tmp_path / "config.json"
    atomic_write_json(config_path, {"attempt_root": str(tmp_path), "runtime": {"upstream": str(upstream)},
        "models": {"native": {"valid_ids": ids, "class_names": list(names)}}, "scenes": {"scene0056_00": {
            "projection": str(projection_path), "annotations": str(annotation_path), "native_geometry": geometry.to_dict()}}})
    source_path = tmp_path / "N.json"
    atomic_write_json(source_path, sources["N"])
    predictions, identities, files = {}, {}, []
    for method, payload in outputs.items():
        path = save_prediction(payload, tmp_path / "predictions" / method)
        predictions[method] = str(path)
        identities[method] = {"prediction_key": payload.prediction_key, "record_key": payload.record_key}
        files.extend([file_identity(path), file_identity(path.parent / "prediction.npz")])
    lock = {"status": "PREDICTIONS_LOCKED", "scene": "scene0056_00", "inputs": [], "outputs": files,
        "predictions": predictions, "prediction_identities": identities}
    lock["identity"] = canonical_digest(lock)
    data = {"availability": "REUSABLE_VERIFIED_BB00_NATIVE_ANCHOR", "dataset": "Replica", "config": str(config_path),
        "sources": {"N": {"path": str(source_path), "identity": sources["N"]["identity"]}}}
    binding = {"spec": str(SPEC), "cohorts": spec["cohorts"], "output_root": str(tmp_path),
        "repository_root": str(SPEC.parents[2]), "path_map": {}, "scenes": {"scene0056_00": data}}
    binding["identity"] = canonical_digest(binding)
    # This synthetic fixture has no historical parent rows to alias.
    data["parent_readout_receipt"] = str(tmp_path / "parent/receipt.json")
    atomic_write_json(tmp_path / "parent/evaluation_rows.json", {"rows": []})
    data["predictions"] = {}
    binding["identity"] = canonical_digest({k: v for k, v in binding.items() if k != "identity"})
    result = evaluate_scene(binding, "scene0056_00", lock, tmp_path / "scoring")
    assert result["status"] == "COMPLETE" and result["row_count"] == 8
    rows = {row["method"]: row for row in result["rows"]}
    assert rows["CT_A3_ER"]["metrics"]["ap50"] == pytest.approx(1.)
    assert rows["CT_A3_ER"]["metrics"]["miou"] == pytest.approx(1.)
    assert rows["CT_A5_FC_ONLY"]["metrics"]["miou"] == pytest.approx(.5)
    assert rows["CT_A5_FC_ONLY"]["metrics"]["macc"] == pytest.approx(.5)
    check = result["registry_checks"]["CT_A5_FC_ONLY"]
    assert check["mask_registry_owners"] == [7, 9]
    assert check["official_manifest_owners"] == [9]
    assert check["unknown_semantic_positive_owners"] == [7]
    assert check["native_unavailable_added_rows"] == [9]
    assert check["semantic_confusion_count"] == 200
    assert evaluate_scene(binding, "scene0056_00", lock, tmp_path / "scoring")["identity"] == result["identity"]
    from static_ovmap.cvpr_compact.diagnostics import analyze_scoring_comparison
    actual = analyze_scoring_comparison(rows["CT_A1_E"]["evaluation_receipt"],
                                       rows["CT_A3_ER"]["evaluation_receipt"], {9})
    assert actual["added_TP50"] == 1 and actual["added_FP50"] == 0
    assert not actual["added_matching_ambiguous"] and not actual["old_owner_rank_changes"]
    unknown = analyze_scoring_comparison(rows["CT_A1_E"]["evaluation_receipt"],
                                        rows["CT_A5_FC_ONLY"]["evaluation_receipt"], {9})
    assert len(unknown["lost_old_GT_matches50"]) == 1
    assert unknown["old_owner_rank_changes"][0]["owner"] == 7
    assert not unknown["old_owner_rank_changes"][0]["exported_after"]
    from static_ovmap.cvpr_compact.evaluation import trace_class_metrics, released_pool_with_classes
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex
    scoring = json.loads(Path(rows["CT_A5_FC_ONLY"]["evaluation_receipt"]).read_text())
    classes = {r["class_id"]: r for r in trace_class_metrics(scoring)}
    assert classes[first]["iou"] == 0. and classes[second]["iou"] == 1.
    assert classes[first]["ap50"] == 0. and classes[second]["ap50"] == 1.
    assert len(classes) == len(ids) and classes[first]["gt_points"] == 100
    namespace = load_released_module(upstream / "scripts/eval_utils.py")
    namespace["init"]("Replica")
    pooled, details = released_pool_with_classes(tmp_path / "pool", [rows["CT_A5_FC_ONLY"]], namespace,
        ["scene0056_00"], "CT_A5_FC_ONLY", "OFFICIAL_CURRENT_CLASS", ConsumptionIndex())
    assert pooled["metrics"]["miou"] == .5
    assert {r["class_id"]: r for r in details["classes"]}[first]["ap50"] == 0.
    original = namespace["evaluate"]
    namespace["evaluate"] = lambda *args: pytest.fail("complete pool must reuse its captured original class values")
    again, reused = released_pool_with_classes(tmp_path / "pool", [rows["CT_A5_FC_ONLY"]], namespace,
        ["scene0056_00"], "CT_A5_FC_ONLY", "OFFICIAL_CURRENT_CLASS", ConsumptionIndex())
    namespace["evaluate"] = original
    assert again == pooled and reused["identity"] == details["identity"]

    from static_ovmap.cvpr_compact.partial_execution import evaluate_partial_scene
    _, _, _, selected, scope = partial_scope_fixture("scene0056_00")
    selected_ids = {row["id"] for row in selected}
    partial_lock = {**scope, "status": "PARTIAL_PREDICTIONS_LOCKED", "inputs": [], "outputs": files,
        "predictions": {key: value for key, value in predictions.items() if key in selected_ids},
        "prediction_identities": {key: value for key, value in identities.items() if key in selected_ids}}
    partial_lock["identity"] = canonical_digest(partial_lock)
    partial = evaluate_partial_scene(binding, "scene0056_00", partial_lock, tmp_path / "partial_scoring")
    assert partial["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS" and partial["row_count"] == 4
    assert partial["blocked_methods"] == scope["blocked_methods"]
    assert partial["expected_method_ids"] == [row["id"] for row in spec["methods"]]
    assert {row["method"] for row in partial["rows"]} == selected_ids
    for row in partial["rows"]:
        assert row["metrics"] == rows[row["method"]]["metrics"] and row["status"] == "COMPLETE"
        assert partial["registry_checks"][row["method"]]["semantic_confusion_count"] == 200
    assert evaluate_partial_scene(binding, "scene0056_00", partial_lock,
                                  tmp_path / "partial_scoring")["identity"] == partial["identity"]

    from static_ovmap.cvpr_compact.partial_execution import pool_partial_cohort
    from static_ovmap.cvpr_compact import evaluation as compact_evaluation
    from static_ovmap.cvpr_compact import partial_execution
    # Reuse synthetic scoring files to test the full fixed-cohort packaging path.
    pool_binding = {**binding, "scenes": {scene: data for scene in spec["cohorts"]["replica8"]}}
    pool_binding["identity"] = canonical_digest({k: v for k, v in pool_binding.items() if k != "identity"})
    for scene in spec["cohorts"]["replica8"]:
        scene_result = copy.deepcopy(result)
        scene_result["scene"] = scene
        for row in scene_result["rows"]:
            row["scene"] = scene
            row["identity"] = canonical_digest({k: v for k, v in row.items() if k != "identity"})
        if scene == "office1":
            _, _, _, _, scene_scope = partial_scope_fixture(scene)
            scene_result.update(scene_scope, status="PARTIAL_WITH_TECHNICAL_BLOCKS",
                rows=[row for row in scene_result["rows"] if row["method"] in selected_ids])
        scene_result["identity"] = canonical_digest({k: v for k, v in scene_result.items() if k != "identity"})
        atomic_write_json(tmp_path / "evaluation" / scene / "receipt.json", scene_result)
    monkeypatch.setattr(compact_evaluation, "require_frozen_execution", lambda *args: "SYNTHETIC_FIXTURE_ONLY")
    monkeypatch.setattr(partial_execution, "require_frozen_execution", lambda *args: "SYNTHETIC_FIXTURE_ONLY")
    pooled_scope = pool_partial_cohort(pool_binding, "replica8")
    assert pooled_scope["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS"
    assert pooled_scope["complete_pool_count"] == 4 and pooled_scope["required_pool_count"] == 8
    assert set(pooled_scope["methods"]) == selected_ids
    assert len(pooled_scope["blocked_methods"]) == 4
    for method, measured in pooled_scope["methods"].items():
        assert measured["status"] == "COMPLETE" and measured["scene_order"] == spec["cohorts"]["replica8"]
        assert measured["metrics"]["miou"] == rows[method]["metrics"]["miou"]
        assert len(measured["row_identities"]) == 8 and len(measured["ordered_inputs"]) == 8
    assert pool_partial_cohort(pool_binding, "replica8")["identity"] == pooled_scope["identity"]


def test_completed_base_worker_verifies_outputs_without_spending_retry(tmp_path):
    import sys
    from static_ovmap.cvpr_compact import base_sources
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    binding = {"repository_root": str(SPEC.parents[2]), "gpu": "2"}
    job = {"output_root": str(tmp_path)}
    index = ConsumptionIndex()
    producer = index.identity(base_sources.__file__)
    argv = [sys.executable, "-m", "static_ovmap.cvpr_compact.base_sources", "--worker", "existing-fc",
            "--job", str(tmp_path / "job.json")]
    identity = canonical_digest({"job": job, "producer": producer, "argv": argv})
    vector_path = tmp_path / "F.json"
    atomic_write_json(vector_path, {"feature": [1., 0.]})
    receipt = {"status": "COMPLETE", "inputs": [producer], "outputs": [index.identity(vector_path)]}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(tmp_path / "receipt.json", receipt)
    command = {"attempt": 1, "status": "COMPLETE", "argv": argv, "cwd": binding["repository_root"], "exit_code": 0}
    ledger_path = tmp_path / ("command_" + identity + ".json")
    atomic_write_json(ledger_path, {"input_identity": identity, "attempts": [command]})
    reused = base_sources._run_worker(binding, sys.executable, job, "existing-fc", ConsumptionIndex())
    assert reused["reused_successful_command"] is True
    assert json.loads(ledger_path.read_text())["attempts"] == [command]
    atomic_write_json(vector_path, {"feature": [0., 1.]})
    with pytest.raises(ValueError, match="changed|identity|hash"):
        base_sources._run_worker(binding, sys.executable, job, "existing-fc", ConsumptionIndex())


def test_shared_recovery_export_matches_final_method_outputs_and_current_ranks(tmp_path, monkeypatch):
    from static_ovmap.cvpr_compact.recovery_run import RecoveryInputs, export_conditions
    from static_ovmap.cvpr_compact.outputs import build_method_outputs
    from static_ovmap.module_validation.scannet_study import load_prediction
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    xyz = np.column_stack((np.arange(6), np.zeros((6, 2)))).astype(np.float32)
    raw, painted = np.array([7, 7, 9, 9, 9, 9]), np.array([7, 7, 0, 0, 0, 0])
    geometry = GeometryIdentity(_array_digest(xyz), "2" * 64, "3" * 64, "projection", 6)
    baseline = PredictionPayload("N0", "N0", "scene0056_00", geometry, painted,
                                 np.array([1, 1, 0, 0, 0, 0]), ((7, 1.),), {})
    baseline.lock()
    registry = build_registry(xyz, raw, painted, minimum_rows=2)
    def source(name, owners, labels):
        value = {"source": name, "valid_ids": [1, 2], "objects": {str(owner): {
            "available": owner in labels, "label": labels.get(owner), "scores": ([1., 0.] if labels.get(owner) == 1 else [0., 1.]) if owner in labels else None,
            "attempted_request_ids": [], "used_request_ids": []} for owner in owners}}
        value["identity"] = canonical_digest(value)
        return value
    sources = {"N": source("N", [7], {7: 1}), "Q": source("Q", [7], {}), "F": source("F", [7], {})}
    regions = {"G1": source("G1", [9], {9: 1}), "G3": source("G3", [9], {9: 2}), "U2": source("U2", [9], {})}
    binding = {"spec": str(SPEC), "final_temperatures": {"N": .1, "Q": .1, "F": .1}}
    inputs = RecoveryInputs(scene="scene0056_00", binding_identity="fixture", data={}, baseline=baseline,
        xyz=xyz, faces=np.empty((0, 3), np.int64), raw=raw, sources=sources, valid_ids=[1, 2],
        nearest=np.tile(np.arange(6), 100), matched=np.ones(600, bool), identity="fixture")
    exported = export_conditions(binding, inputs, registry, regions, ("G1", "G3", "U2"), tmp_path, ConsumptionIndex())
    from static_ovmap.cvpr_compact import outputs as output_module
    original_fusion = output_module.fuse_readout
    def backend_ulp_audit(*args, **kwargs):
        labels, audit = original_fusion(*args, **kwargs)
        for row in audit.values():
            if row["probabilities"] is not None:
                row["probabilities"] = np.nextafter(row["probabilities"], np.inf).tolist()
        return labels, audit
    monkeypatch.setattr(output_module, "fuse_readout", backend_ulp_audit)
    recovery = {"G1_FC": regions["G1"], "G3_FC": regions["G3"], "ARCHIVED_U2_FC": regions["U2"],
                "G1_NATIVE": source("G1_NATIVE", [9], {9: 2})}
    expected = build_method_outputs(baseline, raw, registry, sources, binding["final_temperatures"],
        [1, 2], inputs.nearest, inputs.matched, recovery, json.loads(SPEC.read_text())["methods"])
    assert set(exported) == {"CT_A3_ER", "CT_G3", "CT_H_U2"}
    for method, entry in exported.items():
        actual = load_prediction(entry["manifest"])
        np.testing.assert_array_equal(actual.owner_ids, expected[method].owner_ids)
        np.testing.assert_array_equal(actual.semantic_labels, expected[method].semantic_labels)
        assert actual.instance_ranks == expected[method].instance_ranks
        assert actual.record_key == expected[method].record_key
    assert dict(load_prediction(exported["CT_A3_ER"]["manifest"]).instance_ranks)[7] == .5
    assert dict(load_prediction(exported["CT_G3"]["manifest"]).instance_ranks)[7] == 1.


def test_timing_mean_requires_all_24_real_leaves_and_one_physical_gpu():
    from static_ovmap.cvpr_compact.timing import aggregate_timings, PRODUCTION_CALLABLE
    from static_ovmap.cvpr_compact.protocol import experiment_matrix

    spec = json.loads(SPEC.read_text())
    rows = []
    for number, leaf in enumerate(experiment_matrix(spec)["timings"]):
        row = {**leaf, "status": "COMPLETE", "measured_seconds": number + .25,
            "parity": {"status": "PASS"}, "production_callable": PRODUCTION_CALLABLE,
            "resident_model": True, "persistent_feature_cache_enabled": False,
            "persistent_view_cache_enabled": False, "persistent_result_cache_enabled": False,
            "hardware": {"uuid": "one-gpu", "name": "NVIDIA A40", "compute_capability": "8.6"}}
        row["identity"] = canonical_digest(row)
        rows.append(row)
    result = aggregate_timings(spec, list(reversed(rows)))
    for arm in spec["timing"]["arms"]:
        expected = [r["measured_seconds"] for r in rows if r["arm"] == arm]
        assert result["arms"][arm]["scene_count"] == 8
        assert result["arms"][arm]["mean_seconds"] == sum(expected) / 8
    assert result["none_seconds"] == 0 and result["none_basis"] == "NO_RECOVERY_BY_DEFINITION"
    with pytest.raises(ValueError, match="24|complete"):
        aggregate_timings(spec, rows[:-1])
    with pytest.raises(ValueError, match="duplicate|24"):
        aggregate_timings(spec, [*rows[:-1], rows[0]])
    changed = copy.deepcopy(rows)
    changed[-1]["hardware"]["uuid"] = "second-gpu"
    changed[-1]["identity"] = canonical_digest({k: v for k, v in changed[-1].items() if k != "identity"})
    with pytest.raises(ValueError, match="physical GPU"):
        aggregate_timings(spec, changed)
    changed = copy.deepcopy(rows)
    changed[0]["measured_seconds"] = 0.
    changed[0]["identity"] = canonical_digest({k: v for k, v in changed[0].items() if k != "identity"})
    with pytest.raises(ValueError, match="positive|measured"):
        aggregate_timings(spec, changed)


def test_proven_blocked_timing_is_unmeasured_and_cannot_reserve_or_replace_a_call(tmp_path, monkeypatch):
    from static_ovmap.cvpr_compact import timing
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    _, projected, failed, _, scope = partial_scope_fixture()
    index = ConsumptionIndex()
    failed_path, projected_path = tmp_path / "failed.json", tmp_path / "projected.json"
    atomic_write_json(failed_path, failed)
    atomic_write_json(projected_path, projected)
    parent = {"status": "BLOCKED_TECHNICAL_PARENT", "scene": "office1", "arm": "G1_FC",
        "technical_block": scope["blocked_sources"]["G1_FC"],
        "failed_receipt": index.identity(failed_path), "projected_manifest": index.identity(projected_path)}
    parent["identity"] = canonical_digest(parent)
    monkeypatch.setattr(timing, "reserve_measurement", lambda *args: pytest.fail("a blocked leaf must not reserve a call"))
    binding = {"identity": "synthetic-binding", "gpu": "2"}
    path = tmp_path / "cold/receipt.json"
    result = timing.record_blocked_measurement(binding, "office1", "G1_FC", parent, path, index)
    assert result["status"] == "BLOCKED_UNMEASURED" and result["measured_seconds"] is None
    assert result["physical_calls_reserved"] == 0 and result["cold_call_executed"] is False
    assert result["measurement_reserved"] is False and result["parity"] is None
    assert not (path.parent / "call").exists()
    assert timing.record_blocked_measurement(binding, "office1", "G1_FC", parent, path, index) == result
    atomic_write_json(path, {"status": "RESERVED", "input_identity": "observed-call"})
    with pytest.raises(ValueError, match="overwrite|reserved|replay"):
        timing.record_blocked_measurement(binding, "office1", "G1_FC", parent, path, index)


def test_partial_cold_summary_keeps_24_fixed_leaves_and_never_reports_a_seven_scene_mean(tmp_path):
    from static_ovmap.cvpr_compact.timing import aggregate_timings, record_blocked_measurement, PRODUCTION_CALLABLE
    from static_ovmap.cvpr_compact.protocol import experiment_matrix
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    spec, projected, failed, _, scope = partial_scope_fixture()
    index = ConsumptionIndex()
    failed_path, projected_path = tmp_path / "failed.json", tmp_path / "projected.json"
    atomic_write_json(failed_path, failed)
    atomic_write_json(projected_path, projected)
    rows = []
    for number, leaf in enumerate(experiment_matrix(spec)["timings"]):
        scene, arm = leaf["scene"], leaf["arm"]
        if scene == "office1" and arm in ("G1_FC", "G3_FC"):
            parent = {"status": "BLOCKED_TECHNICAL_PARENT", "scene": scene, "arm": arm,
                "technical_block": scope["blocked_sources"][arm],
                "failed_receipt": index.identity(failed_path), "projected_manifest": index.identity(projected_path)}
            parent["identity"] = canonical_digest(parent)
            row = record_blocked_measurement({"identity": "synthetic-binding", "gpu": "2"}, scene, arm,
                parent, tmp_path / scene / arm / "receipt.json", index)
        else:
            row = {**leaf, "status": "COMPLETE", "measured_seconds": number + .25,
                "parity": {"status": "PASS"}, "production_callable": PRODUCTION_CALLABLE,
                "resident_model": True, "persistent_feature_cache_enabled": False,
                "persistent_view_cache_enabled": False, "persistent_result_cache_enabled": False,
                "hardware": {"uuid": "one-gpu", "name": "NVIDIA A40", "compute_capability": "8.6"}}
            row["identity"] = canonical_digest(row)
        rows.append(row)
    result = aggregate_timings(spec, list(reversed(rows)))
    assert result["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS"
    assert result["leaf_count"] == 22 and result["planned_leaf_count"] == 24 and result["blocked_leaf_count"] == 2
    for arm in ("G1_FC", "G3_FC"):
        summary = result["arms"][arm]
        assert summary["mean_seconds"] is None and summary["sum_seconds"] is None
        assert summary["scene_count"] == 8 and summary["complete_scene_count"] == 7
        assert summary["scene_order"] == spec["cohorts"]["replica8"] and len(summary["receipt_identities"]) == 8
    expected = [row["measured_seconds"] for row in rows if row["arm"] == "ARCHIVED_U2_FC"]
    assert result["arms"]["ARCHIVED_U2_FC"]["mean_seconds"] == sum(expected) / 8
    with pytest.raises(ValueError, match="24|fixed"):
        aggregate_timings(spec, rows[:-1])
    changed = copy.deepcopy(rows)
    blocked = next(row for row in changed if row["status"] == "BLOCKED_UNMEASURED")
    blocked["measured_seconds"] = 0.
    blocked["identity"] = canonical_digest({k: v for k, v in blocked.items() if k != "identity"})
    with pytest.raises(ValueError, match="unmeasured|block"):
        aggregate_timings(spec, changed)
    changed = copy.deepcopy(rows)
    blocked = next(row for row in changed if row["status"] == "BLOCKED_UNMEASURED")
    blocked["technical_block"]["GT_input"] = True
    blocked["technical_block"]["identity"] = canonical_digest({
        k: v for k, v in blocked["technical_block"].items() if k != "identity"})
    blocked["identity"] = canonical_digest({k: v for k, v in blocked.items() if k != "identity"})
    with pytest.raises(ValueError, match="unmeasured|block"):
        aggregate_timings(spec, changed)


def test_cold_controller_calls_22_independent_arms_once_and_reuses_complete_measurements(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import torch
    from static_ovmap.cvpr_compact import timing
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    spec, projected, failed, _, scope = partial_scope_fixture()
    binding = {"identity": "synthetic-binding", "gpu": "2", "spec": str(SPEC), "output_root": str(tmp_path)}
    index = ConsumptionIndex()
    failed_path, projected_path = tmp_path / "failed.json", tmp_path / "projected.json"
    atomic_write_json(failed_path, failed)
    atomic_write_json(projected_path, projected)
    parents, documents = {}, {}
    for scene in spec["cohorts"]["replica8"]:
        parents[scene], documents[scene] = {}, {}
        for arm in spec["timing"]["arms"]:
            if scene == "office1" and arm in ("G1_FC", "G3_FC"):
                entry = {"status": "BLOCKED_TECHNICAL_PARENT", "scene": scene, "arm": arm,
                    "technical_block": scope["blocked_sources"][arm],
                    "failed_receipt": index.identity(failed_path), "projected_manifest": index.identity(projected_path)}
                entry["identity"] = canonical_digest(entry)
                parents[scene][arm], documents[scene][arm] = entry, None
            else:
                documents[scene][arm] = {"identity": canonical_digest([scene, arm]), "scene": scene}
    monkeypatch.setattr(timing, "_preflight_parents", lambda *args: (parents, documents))
    monkeypatch.setattr(timing, "require_frozen_execution", lambda *args: "SYNTHETIC_FIXTURE_ONLY")
    monkeypatch.setattr(timing, "assert_serial_execution", lambda *args: None)
    hardware = {"uuid": "one-gpu", "name": "NVIDIA A40", "compute_capability": "8.6"}
    monkeypatch.setattr(timing, "_hardware", lambda *args: hardware)
    monkeypatch.setattr(timing, "load_recovery_inputs", lambda binding, scene, **kwargs:
        SimpleNamespace(identity=scene, scene=scene, data={}))
    @contextmanager
    def lease(*args):
        yield SimpleNamespace(active=True)
    monkeypatch.setattr(timing, "gpu_lease", lease)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    loads, calls = [], []
    class Session:
        def __init__(self, binding, data, index, *, cache):
            assert cache is None
            self.model_key, self.text_identity, self.ids = "model", {"sha256": "a" * 64}, [1]
            self.model_load_seconds = .01
        def load_model(self):
            loads.append(1)
            self.model, self.weight_audit = object(), {"strict": True}
    monkeypatch.setattr(timing, "FCSession", Session)
    def recover(binding, inputs, output, *, arms, session, cold, lease):
        assert cold is True and len(arms) == 1 and lease.active
        calls.append((inputs.scene, arms[0]))
        value = {"status": "COMPLETE", "scene": inputs.scene, "candidate_count": 0,
            "source_available_additions": {arms[0]: 0}, "physical_image_encodings": 0,
            "physical_region_poolings": 0, "outputs": []}
        value["identity"] = canonical_digest(value)
        atomic_write_json(Path(output) / "receipt.json", value)
        return value
    monkeypatch.setattr(timing, "recover_fc", recover)
    def finish(row, path, scientific, index):
        assert scientific is not None
        row.update(status="COMPLETE", parity={"status": "PASS"})
        row["identity"] = canonical_digest({key: value for key, value in row.items() if key != "identity"})
        atomic_write_json(path, row)
    monkeypatch.setattr(timing, "_finish_parity", finish)
    first = timing.run_timings(binding, {})
    assert len(calls) == len(set(calls)) == 22 and len(loads) == 1
    assert ("office1", "U2") in calls and ("office1", "G1") not in calls and ("office1", "G3") not in calls
    assert first["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS" and first["planned_leaf_count"] == 24
    hashes = {path: sha256_file(path) for path in (tmp_path / "timing").glob("*/CT_*/receipt.json")}
    assert len(hashes) == 0  # Arms use source names, not method IDs.
    hashes = {path: sha256_file(path) for path in (tmp_path / "timing").glob("*/*/receipt.json")}
    assert len(hashes) == 24
    timing.run_timings(binding, {})
    assert len(calls) == 22 and len(loads) == 1
    assert all(sha256_file(path) == digest for path, digest in hashes.items())


def test_cold_parity_checks_selected_features_support_labels_and_current_ranks(tmp_path):
    from static_ovmap.cvpr_compact.timing import compare_recovery_parity
    from static_ovmap.module_validation.scannet_study import save_prediction
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    xyz = np.zeros((4, 3), np.float32)
    geometry = GeometryIdentity(_array_digest(xyz), "2" * 64, "3" * 64, "projection", 4)
    def receipt(directory, vectors, labels=(1, 1, 2, 2), ranks=((7, 1.), (9, 1.))):
        index = ConsumptionIndex()
        directory.mkdir()
        source = {"valid_ids": [1, 2], "objects": {"9": {"available": True, "label": int(labels[-1]),
            "attempted_request_ids": ["selected"], "used_request_ids": ["selected"],
            "failed_request_ids": [], "scores": [0., 1.]}}}
        source["identity"] = canonical_digest(source)
        path = directory / "source.json"
        path.write_text(json.dumps(source))
        feature_path = directory / "features.npz"
        _write_npz(feature_path, {"request_ids": np.asarray(list(vectors), dtype="U64"),
                                 "features": np.stack(list(vectors.values())).astype(np.float32)})
        payload = PredictionPayload("CT_A3_ER", "COMBO", "office0", geometry, np.array([7, 7, 9, 9]),
            np.array(labels), ranks, {})
        payload.lock()
        prediction_path = save_prediction(payload, directory / "prediction")
        manifest = json.loads(prediction_path.read_text())
        value = {"status": "COMPLETE", "scene": "office0", "resident_input_identity": "same-base",
            "registry_identity": "same-registry", "sources": {"G1": index.identity(path)},
            "plan": {"G1": {"9": ["selected"]}}, "features": index.identity(feature_path),
            "exports": {"CT_A3_ER": {"manifest": str(prediction_path), "files": [index.identity(prediction_path),
                index.identity(prediction_path.parent / manifest["arrays"]["path"], manifest["arrays"])]}}}
        value["identity"] = canonical_digest(value)
        return value
    science = receipt(tmp_path / "science", {"selected": np.array([0., 1.]), "other-arm": np.array([1., 0.])})
    cold = receipt(tmp_path / "cold", {"selected": np.array([0., 1. + 1e-6])})
    result = compare_recovery_parity(science, cold, "G1")
    assert result["status"] == "PASS" and result["compared_request_count"] == 1
    assert result["feature_tolerance"] == {"atol": 1e-5, "rtol": 1e-5, "dtype": "float32"}
    assert compare_recovery_parity(science, cold, "G1_FC")["status"] == "PASS"
    changed = receipt(tmp_path / "wrong-features", {"selected": np.array([1., 0.])})
    with pytest.raises(ValueError, match="feature"):
        compare_recovery_parity(science, changed, "G1")
    changed = receipt(tmp_path / "wrong-labels", {"selected": np.array([0., 1.])}, labels=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="label|semantic"):
        compare_recovery_parity(science, changed, "G1")
    changed = receipt(tmp_path / "wrong-ranks", {"selected": np.array([0., 1.])}, ranks=((7, .5), (9, 1.)))
    with pytest.raises(ValueError, match="rank"):
        compare_recovery_parity(science, changed, "G1")


def test_cold_measurement_reservation_never_replays_an_observed_or_lost_call(tmp_path):
    from static_ovmap.cvpr_compact.timing import reserve_measurement

    path = tmp_path / "receipt.json"
    row = reserve_measurement(path, "binding", "office0", "G1", "input")
    assert row["status"] == "RESERVED" and row["physical_calls_reserved"] == 1
    with pytest.raises(RuntimeError, match="already reserved|replay"):
        reserve_measurement(path, "binding", "office0", "G1", "input")
    row.update(status="MEASURED", measured_seconds=.01)
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="already reserved|replay"):
        reserve_measurement(path, "binding", "office0", "G1", "changed-input")


def test_shared_production_cold_call_rebuilds_views_and_matches_warm_exports(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from static_ovmap.cvpr_compact import recovery_run
    from static_ovmap.cvpr_compact.timing import compare_recovery_parity

    xyz, faces, raw = plane(owner=9)
    xyz, raw = np.tile(xyz, (100, 1)), np.tile(raw, 100)
    _write_npz(tmp_path / "surface.npz", {"surface_xyz": xyz, "surface_faces": faces, "original_owner": raw})
    image = np.zeros((20, 20, 3), np.uint8)
    assert cv2.imwrite(str(tmp_path / "rgb.png"), image)
    _write_npz(tmp_path / "depth.npz", {"depth_m": np.full((20, 20), 2., np.float32)})
    K, pose = camera()
    frame = {"frame_id": 1, "image_size_hw": [20, 20], "intrinsics": K.tolist(), "pose_c2w": pose.tolist(),
        "rgb_path": "rgb.png", "rgb_sha256": sha256_file(tmp_path / "rgb.png"),
        "depth_path": "depth.npz", "depth_sha256": sha256_file(tmp_path / "depth.npz")}
    capture = {"scene_id": "scene0056_00", "scheduled_frame_ids": [1], "completed_frame_ids": [1],
        "frames": [frame], "surface": {"path": "surface.npz", "sha256": sha256_file(tmp_path / "surface.npz")}}
    capture["identity"] = canonical_digest(capture)
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture))
    geometry = GeometryIdentity(_array_digest(xyz), _array_digest(faces), "1" * 64, "projection", len(raw))
    native = PredictionPayload("N0", "N0", "scene0056_00", geometry, np.zeros(len(raw), np.int64),
                               np.zeros(len(raw), np.int64), (), {})
    native.lock()
    sources = {}
    for name in ("N", "Q", "F"):
        source = {"source": name, "objects": {}, "valid_ids": [1, 2]}
        source["identity"] = canonical_digest(source)
        sources[name] = source
    data = {"capture_manifest": str(capture_path), "FC_physical_model_identity": "model",
            "FC_text": {"sha256": "text"}}
    binding = {"identity": "binding", "spec": str(SPEC), "output_root": str(tmp_path / "output"),
        "gpu_lock": "synthetic-lock", "final_temperatures": {"N": .1, "Q": .1, "F": .1}}
    inputs = recovery_run.RecoveryInputs("scene0056_00", "binding", data, native, xyz, faces, raw,
        sources, [1, 2], np.arange(len(raw)), np.ones(len(raw), bool), "resident")
    monkeypatch.setattr(recovery_run, "require_frozen_execution", lambda binding, scene: "SYNTHETIC_ONLY")
    projected_roots = []
    original_projector = recovery_run.build_projected_views
    def projector(capture, native, root, **kwargs):
        projected_roots.append(Path(root))
        return original_projector(capture, native, root, **kwargs)
    monkeypatch.setattr(recovery_run, "build_projected_views", projector)
    encoded = []
    def encode(requests, loaders):
        features = {}
        for rid in requests:
            value = loaders[rid](rid)
            assert value["target"].sum() == 400
            features[rid] = np.array([0., 1.], np.float32)
            encoded.append(rid)
        return features, {"physical_image_encodings": 1, "physical_region_poolings": len(features),
                          "persistent_feature_cache_enabled": False}
    session = SimpleNamespace(cache=None, model=object(), model_key="model", ids=[1, 2],
        text_identity={"sha256": "text"}, text=np.eye(2, dtype=np.float32), device="cuda",
        model_load_seconds=0., encode=encode)
    lease = recovery_run.GPULease("binding", "synthetic-lock", os.getpid())
    warm = recovery_run.recover_fc(binding, inputs, tmp_path / "warm", arms=("G1",), session=session, lease=lease)
    cold = recovery_run.recover_fc(binding, inputs, tmp_path / "cold", arms=("G1",), session=session, lease=lease, cold=True)
    assert projected_roots == [tmp_path / "output/projected_views/scene0056_00", tmp_path / "cold/views"]
    assert len(encoded) == 2 and encoded[0] == encoded[1]
    assert warm["candidate_count"] == cold["candidate_count"] == 1
    assert cold["source_available_additions"] == {"G1": 1}
    assert compare_recovery_parity(warm, cold, "G1")["status"] == "PASS"
    with pytest.raises(ValueError, match="fresh isolated"):
        recovery_run.recover_fc(binding, inputs, tmp_path / "cold", arms=("G1",), session=session, lease=lease, cold=True)
    def unavailable(requests, loaders):
        return {}, {"physical_image_encodings": 1, "physical_region_poolings": 0,
            "persistent_feature_cache_enabled": False, "requests": {rid: {
                "status": "UNAVAILABLE_TECHNICAL_FAILURE", "reason": "EMPTY_DENSE_MASK_SUPPORT"} for rid in requests}}
    session.encode = unavailable
    for name, is_cold in (("unavailable-warm", False), ("unavailable-cold", True)):
        with pytest.raises(RuntimeError, match="ALL_SEMANTIC_REQUESTS_FAILED"):
            recovery_run.recover_fc(binding, inputs, tmp_path / name, arms=("G1",),
                                   session=session, lease=lease, cold=is_cold)
        failed = json.loads((tmp_path / name / "receipt.json").read_text())
        assert failed["status"] == "FAILED" and failed["physical_image_encodings"] == 1
        assert "source_available_additions" not in failed
    session.encode = lambda requests, loaders: ({}, {"requests": {}})
    with pytest.raises(RuntimeError, match="no recorded technical failure"):
        recovery_run.recover_fc(binding, inputs, tmp_path / "missing-evidence", arms=("G1",), session=session, lease=lease)


def test_pipeline_prediction_phase_gates_before_any_main_worker(tmp_path, monkeypatch):
    from static_ovmap.cvpr_compact import workflow

    spec = json.loads(SPEC.read_text())
    binding = {"spec": str(SPEC), "cohorts": spec["cohorts"], "output_root": str(tmp_path / "output"),
        "repository_root": str(tmp_path / "repository"), "gpu": "2", "scenes": {spec["smoke_scene"]: {
            "availability": "REUSABLE_VERIFIED_BB00_NATIVE_ANCHOR"}}}
    binding["identity"] = canonical_digest(binding)
    monkeypatch.setattr(workflow, "bind_inputs", lambda *args, **kwargs: binding)
    def forbidden(*args, **kwargs):
        raise AssertionError("main worker was entered before the complete committed freeze")
    monkeypatch.setattr(workflow, "run_worker", forbidden)
    with pytest.raises(RuntimeError, match="freeze"):
        workflow.run_pipeline(SPEC, tmp_path / "repository", phase="predict", resume=True)
    assert not (tmp_path / "output/predictions").exists()


def test_pipeline_completed_worker_checks_receipts_and_does_not_spend_retry(tmp_path, monkeypatch):
    from static_ovmap.cvpr_compact import workflow
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    output = tmp_path / "value.txt"
    output.write_text("complete-output")
    index = ConsumptionIndex()
    result = {"status": "COMPLETE", "inputs": [], "outputs": [index.identity(output)]}
    result["identity"] = canonical_digest(result)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(result))
    binding = {"identity": "binding", "output_root": str(tmp_path / "attempt"), "gpu": "2",
               "repository_root": str(Path(__file__).resolve().parents[2])}
    executions = []
    def execute(argv, cwd, log, *, env, input_identity):
        row = {"status": "COMPLETE", "argv": argv, "cwd": cwd, "exit_code": 0}
        ledger = {"input_identity": input_identity, "attempts": [row]}
        ledger_path = Path(log).with_name("command_" + input_identity + ".json")
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(json.dumps(ledger))
        executions.append((argv, env))
        return row
    monkeypatch.setattr(workflow, "execute_leaf", execute)
    kwargs = dict(module="evaluation", python="python", arguments=["--scene", "office0"],
                  result_path=path, expected_status="COMPLETE", inputs=[])
    first = workflow.run_worker(binding, **kwargs)
    second = workflow.run_worker(binding, **kwargs)
    assert first["identity"] == second["identity"] == result["identity"]
    assert len(executions) == 1
    assert executions[0][1]["CUDA_VISIBLE_DEVICES"] == ""
    assert executions[0][1]["OMP_NUM_THREADS"] == "4"
    output.write_text("tampered-output")
    with pytest.raises(ValueError):
        workflow.run_worker(binding, **kwargs)


def test_table3_coverage_sums_counts_instead_of_averaging_scene_ratios():
    from static_ovmap.cvpr_compact.diagnostics import aggregate_recovery_diagnostics, TABLE3_METHODS

    spec = json.loads(SPEC.read_text())
    receipts = []
    for position, scene in enumerate(spec["cohorts"]["replica8"]):
        count = position + 1
        conditions = {}
        for arm, method in TABLE3_METHODS.items():
            conditions[method] = {"source_available_additions": 0 if arm == "NONE" else 1,
                "target_min100_additions": 0, "target_small_additions": 0 if arm == "NONE" else 1,
                "added_TP50": 0, "added_FP50": 0, "ambiguous_added_TP50": int(arm == "G1" and position == 0),
                "ambiguous_added_FP50": 0}
        row = {"status": "COMPLETE", "scene": scene, "candidate_count": count, "conditions": conditions}
        row["identity"] = canonical_digest(row)
        receipts.append(row)
    result = aggregate_recovery_diagnostics(spec, list(reversed(receipts)))
    assert result["arms"]["G1"]["n"] == 8 and result["arms"]["G1"]["N"] == 36
    assert result["arms"]["G1"]["recovered_fraction"] == 8 / 36
    assert result["arms"]["G1"]["target_small_additions"] == 8
    assert result["arms"]["G1"]["matching_ambiguity_flag"]
    assert result["arms"]["NONE"]["n"] == 0 and result["arms"]["NONE"]["N"] == 36
    with pytest.raises(ValueError, match="complete|eight"):
        aggregate_recovery_diagnostics(spec, receipts[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_recovery_diagnostics(spec, [*receipts[:-1], receipts[0]])


def test_external_pdf_table_keeps_columns_and_rejects_incomplete_or_disagreeing_sources():
    from static_ovmap.cvpr_compact.external import parse_table3, reconcile_tables

    text = "Method Online mIoU mAcc AP25 AP50 APall\n" + "\n".join([
        "Mask3D+OM3D x 18.7 29.4 6.8 4.5 3.2",
        "Segment3D+OM3D x 17.3 30.9 19.9 15.2 9.0",
        "OVO-SLAM y 24.9 34.0 28.1 17.5 9.1",
        "OVO-SLAM (30 fps) y 21.8 27.5 21.5 15.2 8.1",
        "Mask3D+OM3D x 8.6 17.5 10.4 8.0 5.1",
        "Segment3D+OM3D x 4.5 13.5 5.0 3.9 2.3",
        "OVO-SLAM y 14.6 27.8 19.4 12.6 5.5",
    ]) + "\nTable 3. Comparison"
    rows = parse_table3(text)
    assert rows[0]["replica8"]["ap50"] == 4.5
    assert rows[0]["scannet_cf18"]["miou"] == 8.6
    assert rows[2]["scannet_cf18"]["apall"] == 5.5
    from static_ovmap.cvpr_compact.protocol import PACKAGE
    supplied = json.loads((PACKAGE / "literature_reference.json").read_text())
    original = copy.deepcopy(supplied)
    audit = reconcile_tables(supplied, rows, copy.deepcopy(rows))
    assert len(audit["differences_from_provided_transcription"]) == 7
    assert audit["rows"] == rows and supplied == original
    assert not audit["full_scorer_protocol_independently_verified"]
    with pytest.raises(ValueError, match="six|row"):
        parse_table3(text.replace("Mask3D+OM3D x 8.6 17.5 10.4 8.0 5.1", "missing"))
    other = copy.deepcopy(rows)
    other[0]["replica8"]["ap50"] = 14.5
    with pytest.raises(ValueError, match="disagree"):
        reconcile_tables(supplied, rows, other)


def _table_inputs(tmp_path):
    from static_ovmap.cvpr_compact.protocol import load_spec, PACKAGE
    from static_ovmap.cvpr_compact.diagnostics import TABLE3_METHODS
    from static_ovmap.cvpr_compact.external import reconcile_tables
    from static_ovmap.cvpr_compact.evaluation import METRICS

    spec = load_spec(SPEC)
    scenes, pools = {}, {}
    for cohort, ordered in spec["cohorts"].items():
        methods = [r["id"] for r in spec["methods"] if cohort in r["cohorts"]]
        for scene in ordered:
            rows = []
            for i, method in enumerate(methods):
                metrics = {name: .1 + .01 * i + .001 * j for j, name in enumerate(METRICS)}
                if method == "CT_A4_NATIVE_RECOVERY" and cohort == "scannet_cf18":
                    metrics["miou"] = None
                row = {"status": "COMPLETE", "scene": scene, "cohort": cohort, "method": method,
                    "rank_mode": "OFFICIAL_CURRENT_CLASS", "metrics": metrics, "metric_unit": "FRACTION",
                    "runtime_overlaps": list(np.r_[np.arange(.5, .95, .05), .25]),
                    "evaluation_identity": canonical_digest([scene, method]),
                    "scorer_context_identity": "synthetic-scoring-context"}
                row["identity"] = canonical_digest(row)
                rows.append(row)
            receipt = {"status": "COMPLETE", "scene": scene, "cohort": cohort, "rows": rows}
            receipt["identity"] = canonical_digest(receipt)
            scenes[scene] = receipt
        for method in methods:
            selected = [next(r for r in scenes[s]["rows"] if r["method"] == method) for s in ordered]
            pooled = {"status": "COMPLETE", "method": method, "cohort": cohort,
                "rank_mode": "OFFICIAL_CURRENT_CLASS", "aggregation": "RELEASED_DATASET_POOL",
                "scene_order": ordered, "metrics": selected[0]["metrics"], "metric_unit": "FRACTION",
                "row_identities": [r["identity"] for r in selected],
                "ordered_inputs": [r["evaluation_identity"] for r in selected],
                "runtime_overlaps": selected[0]["runtime_overlaps"],
                "released_evaluator": {"sha256": "synthetic-evaluator"}}
            pooled["identity"] = canonical_digest(pooled)
            pools[(cohort, method)] = pooled
    supplied = json.loads((PACKAGE / "literature_reference.json").read_text())
    reference = reconcile_tables(supplied, supplied["rows"], supplied["rows"])
    reference["identity"] = canonical_digest(reference)
    arms = {arm: {"method": method, "n": 0 if arm == "NONE" else 8, "N": 36,
        "added_TP50": 0 if arm == "NONE" else 3, "added_FP50": 0 if arm == "NONE" else 2,
        "ambiguous_added_TP50": int(arm == "G1"), "ambiguous_added_FP50": 0,
        "matching_ambiguity_flag": arm == "G1", "scene_order": spec["cohorts"]["replica8"],
        "scene_receipt_identities": ["diagnostic-" + s for s in spec["cohorts"]["replica8"]]}
        for arm, method in TABLE3_METHODS.items()}
    diagnostic = {"status": "COMPLETE", "scene_count": 8, "cohort": "replica8", "arms": arms,
        "common_candidate_denominator": 36}
    diagnostic["identity"] = canonical_digest(diagnostic)
    timing = {"status": "COMPLETE", "leaf_count": 24, "none_seconds": 0.,
        "arms": {arm: {"mean_seconds": .5 + i, "scene_count": 8,
            "scene_order": spec["cohorts"]["replica8"], "receipt_identities": [arm + s for s in spec["cohorts"]["replica8"]]}
            for i, arm in enumerate(spec["timing"]["arms"])}}
    timing["identity"] = canonical_digest(timing)
    return spec, scenes, pools, reference, diagnostic, timing


def test_table_cells_share_pool_provenance_preserve_precision_and_typed_counts(tmp_path):
    from static_ovmap.cvpr_compact.tables import result_store, main_tables, render_tables, format_cell

    spec, scenes, pools, reference, diagnostic, timing = _table_inputs(tmp_path)
    store = result_store(spec, scenes, pools, tmp_path)
    assert len(store["scene_metrics"]) == 172 and len(store["pooled_metrics"]) == 14
    tables = main_tables(spec, store, reference, diagnostic, timing, tmp_path)
    assert [len(tables["tables"][name]) for name in ("table1", "table2", "table3")] == [6, 6, 4]
    cells = [c for rows in tables["tables"].values() for row in rows for c in row["cells"]]
    repeated = [c for c in cells if c["method_id"] == "CT_A3_ER"
                and c["cohort"] == "replica8" and c["metric"] == "apall"]
    assert len(repeated) == 3
    assert len({(c["value_fraction"], c["source_id"], c["receipt_identity"]) for c in repeated}) == 1
    assert format_cell(tables["tables"]["table1"][0]["cells"][0]) == "3.2"
    assert format_cell(repeated[0]) == "13.00"
    ratio = next(c for c in tables["tables"]["table3"][2]["cells"] if c["column_id"] == "recovered_n_over_N")
    assert ratio["numerator"] == 8 and ratio["denominator"] == 36 and format_cell(ratio) == "$8/36$"
    tp = next(c for c in tables["tables"]["table3"][2]["cells"] if c["column_id"] == "added_tp50_entries")
    assert tp["value_count"] == 3 and "dagger" in format_cell(tp)
    rendered = render_tables(tables)
    assert "All result cells are placeholders" not in rendered["table_layout_preview.tex"]
    assert "A4: native recovery & D2 & G1-Native" in rendered["table2_ablation.tex"]
    assert "13.00" in rendered["table1_main.tex"]
    assert "undefined" in rendered["table2_ablation.tex"].lower()
    assert "\\fontsize{9}{10.5}" in rendered["table3_recovery.tex"]


def test_result_store_rejects_missing_or_misaligned_full_cohort_receipts(tmp_path):
    from static_ovmap.cvpr_compact.tables import result_store

    spec, scenes, pools, *_ = _table_inputs(tmp_path)
    with pytest.raises(ValueError, match="complete|fixed"):
        result_store(spec, {s: r for s, r in scenes.items() if s != "room2"}, pools, tmp_path)
    broken = copy.deepcopy(pools)
    row = broken[("replica8", "CT_A3_ER")]
    row["ordered_inputs"] = list(reversed(row["ordered_inputs"]))
    row["identity"] = canonical_digest({k: v for k, v in row.items() if k != "identity"})
    with pytest.raises(ValueError, match="order|input"):
        result_store(spec, scenes, broken, tmp_path)


def _partial_table_inputs(tmp_path):
    spec, scenes, pools, reference, diagnosis, timing = _table_inputs(tmp_path)
    _, _, _, _, scope = partial_scope_fixture()
    blocked = set(scope["blocked_methods"])
    scene = scenes["office1"]
    scene.update(scope, status="PARTIAL_WITH_TECHNICAL_BLOCKS")
    scene["rows"] = [row for row in scene["rows"] if row["method"] not in blocked]
    scene["identity"] = canonical_digest({key: value for key, value in scene.items() if key != "identity"})
    for method in blocked:
        del pools[("replica8", method)]
    for method, pool in pools.items():
        if method[0] == "replica8":
            selected = [next(row for row in scenes[s]["rows"] if row["method"] == method[1])
                        for s in spec["cohorts"]["replica8"]]
            pool["row_identities"] = [row["identity"] for row in selected]
            pool["identity"] = canonical_digest({key: value for key, value in pool.items() if key != "identity"})
    diagnosis["status"] = "PARTIAL_WITH_TECHNICAL_BLOCKS"
    for arm in ("G1", "G3"):
        row = diagnosis["arms"][arm]
        row.update(status="BLOCKED_TECHNICAL", n=None, added_TP50=None, added_FP50=None,
                   ambiguous_added_TP50=None, ambiguous_added_FP50=None, matching_ambiguity_flag=None,
                   complete_scene_count=7, blocked_scenes=["office1"],
                   unavailable_reason="FULL_COHORT_DIAGNOSTIC_HAS_BLOCKED_CONDITIONS")
    diagnosis["identity"] = canonical_digest({key: value for key, value in diagnosis.items() if key != "identity"})
    timing.update(status="PARTIAL_WITH_TECHNICAL_BLOCKS", leaf_count=22, planned_leaf_count=24, blocked_leaf_count=2)
    for arm in ("G1_FC", "G3_FC"):
        timing["arms"][arm].update(mean_seconds=None, complete_scene_count=7, blocked_scene_count=1,
            unavailable_reason="FULL_EIGHT_SCENE_MEAN_HAS_BLOCKED_UNMEASURED_LEAVES")
    timing["identity"] = canonical_digest({key: value for key, value in timing.items() if key != "identity"})
    return spec, scenes, pools, reference, diagnosis, timing


def test_blocked_result_store_keeps_fixed_slots_and_same_na_provenance_in_all_tables(tmp_path):
    from static_ovmap.cvpr_compact.tables import result_store, main_tables, format_cell, render_tables

    spec, scenes, pools, reference, diagnosis, timing = _partial_table_inputs(tmp_path)
    store = result_store(spec, scenes, pools, tmp_path)
    assert store["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS"
    assert len(store["scene_metrics"]) == 172 and len(store["pooled_metrics"]) == 14
    assert store["main_scene_outputs"] == {"complete": 168, "blocked": 4, "required": 172}
    assert store["internal_pools"] == {"complete": 10, "blocked": 4, "required": 14}
    blocked = [row for row in store["pooled_metrics"] if row["source_kind"] == "UNAVAILABLE"]
    assert len(blocked) == 4
    assert all(row["completed_coverage"] == 7 and row["required_coverage"] == 8 for row in blocked)
    assert all(all(value is None for value in row["metrics"].values()) for row in blocked)
    tables = main_tables(spec, store, reference, diagnosis, timing, tmp_path)
    assert tables["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS"
    assert [len(tables["tables"][name]) for name in ("table1", "table2", "table3")] == [6, 6, 4]
    cells = [cell for rows in tables["tables"].values() for row in rows for cell in row["cells"]]
    repeated = [cell for cell in cells if cell["method_id"] == "CT_A3_ER"
                and cell["cohort"] == "replica8" and cell["metric"] == "apall"]
    assert len(repeated) == 3 and all(format_cell(cell) == "--" for cell in repeated)
    assert len({(cell["source_id"], cell["receipt_identity"]) for cell in repeated}) == 1
    assert all(cell["unavailable_reason"] == "FULL_COHORT_POOL_HAS_BLOCKED_CONDITIONS" for cell in repeated)
    g1 = tables["tables"]["table3"][2]
    assert [format_cell(cell) for cell in g1["cells"]] == ["1", "--", "--", "--", "--", "--", "--"]
    assert "office1" in render_tables(tables)["table3_recovery.tex"]
    missing = dict(pools)
    del missing[("replica8", "CT_H_U2")]
    with pytest.raises(ValueError, match="complete|fixed|pool"):
        result_store(spec, scenes, missing, tmp_path)
    fabricated = copy.deepcopy(scenes)
    fabricated["office1"]["blocked_sources"]["G1_FC"]["GT_input"] = True
    fabricated["office1"]["identity"] = canonical_digest({
        key: value for key, value in fabricated["office1"].items() if key != "identity"})
    with pytest.raises(ValueError, match="block|identity|content changed"):
        result_store(spec, fabricated, pools, tmp_path)


def test_partial_diagnosis_keeps_common_denominator_and_no_partial_sum_as_full_result():
    from static_ovmap.cvpr_compact.diagnostics import aggregate_recovery_diagnostics

    spec, _, _, methods, scope = partial_scope_fixture()
    receipts = []
    for scene in spec["cohorts"]["replica8"]:
        conditions = {row["id"]: {"source_available_additions": int(row["recovery"] != "NONE"),
            "target_min100_additions": int(row["recovery"] != "NONE"), "target_small_additions": 0,
            "added_TP50": int(row["recovery"] != "NONE"), "added_FP50": 0,
            "ambiguous_added_TP50": 0, "ambiguous_added_FP50": 0} for row in spec["methods"]}
        receipt = {"status": "COMPLETE", "scene": scene, "candidate_count": 2, "conditions": conditions}
        if scene == "office1":
            receipt.update(scope, status="PARTIAL_WITH_TECHNICAL_BLOCKS")
            receipt["conditions"] = {row["id"]: conditions[row["id"]] for row in methods}
        receipt["identity"] = canonical_digest(receipt)
        receipts.append(receipt)
    result = aggregate_recovery_diagnostics(spec, receipts)
    assert result["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS" and result["common_candidate_denominator"] == 16
    assert result["arms"]["U2"]["n"] == 8 and result["arms"]["U2"]["complete_scene_count"] == 8
    for arm in ("G1", "G3"):
        row = result["arms"][arm]
        assert row["n"] is None and row["N"] == 16 and row["added_TP50"] is None
        assert row["complete_scene_count"] == 7 and row["blocked_scenes"] == ["office1"]
        assert row["known_measured_counts"]["source_available_additions"] == 7


def test_common_success_native_fc_comparison_uses_saved_same_view_scores_only():
    from static_ovmap.cvpr_compact.diagnostics import common_success_comparison

    fc = {"valid_ids": [1, 2], "objects": {
        "7": {"available": True, "label": 1, "scores": [.9, .1], "used_request_ids": ["same-g1"]},
        "9": {"available": False, "label": 0, "scores": None, "used_request_ids": []}}}
    native = {"valid_ids": [1, 2], "objects": {
        "7": {"available": True, "label": 2, "scores": [.2, .8], "used_request_ids": ["same-g1"]},
        "9": {"available": True, "label": 1, "scores": [.6, .4], "used_request_ids": ["other-g1"]}}}
    result = common_success_comparison(fc, native)
    assert result["common_success_owners"] == [7] and result["native_only_success_owners"] == [9]
    assert result["label_agreement_count"] == 0 and result["cross_encoder_scores_commensurate"] is False
    assert result["objects"][0]["FC_scores"] == [.9, .1] and result["objects"][0]["Native_scores"] == [.2, .8]
    assert result["new_model_inference"] == 0 and result["new_benchmark_row"] is False
    native["objects"]["7"]["used_request_ids"] = ["different-frame"]
    with pytest.raises(ValueError, match="same|view"):
        common_success_comparison(fc, native)


def test_fixed_timing_arm_names_resolve_to_the_same_production_recovery_arms():
    from static_ovmap.cvpr_compact.timing import recovery_arm

    spec = json.loads(SPEC.read_text())
    assert [recovery_arm(name) for name in spec["timing"]["arms"]] == ["U2", "G1", "G3"]
    assert [recovery_arm(name) for name in ("U2", "G1", "G3")] == ["U2", "G1", "G3"]
    with pytest.raises(ValueError, match="arm"):
        recovery_arm("G1_NATIVE")


def test_per_class_metrics_keep_gt_present_semantics_and_distinct_instance_vocabulary():
    from static_ovmap.cvpr_compact.evaluation import per_class_metrics

    matrix = np.array([[0, 0, 0, 0], [3, 7, 0, 0], [0, 4, 0, 0], [0, 1, 0, 0]])
    averages = {"classes": {"two": {"ap": 0., "ap50%": 0., "ap25%": 0.},
                            "one": {"ap": .4, "ap50%": .5, "ap25%": .6}}}
    rows = per_class_metrics(averages, matrix, [1, 2, 3], ["one", "two", "three"], [2, 1], ["two", "one"])
    assert rows[0]["iou"] == 7 / 15 and rows[0]["accuracy"] == .7
    assert rows[2]["apall"] is None and rows[2]["instance_unavailable_reason"] == "NOT_IN_RELEASED_INSTANCE_VOCABULARY"
    assert rows[2]["iou"] == 0. and rows[2]["gt_present"]
    matrix[3] = 0
    rows = per_class_metrics(averages, matrix, [1, 2, 3], ["one", "two", "three"], [2, 1], ["two", "one"])
    assert rows[2]["iou"] is None and not rows[2]["gt_present"]


def test_semantic_outcomes_allow_no_views_but_block_all_failed_selected_requests():
    from static_ovmap.cvpr_compact.region_worker import validate_request_outcomes

    validate_request_outcomes({}, {}, {"requests": {}})
    failures = {"requests": {"a": {"status": "UNAVAILABLE_TECHNICAL_FAILURE", "reason": "CUDA_OUT_OF_MEMORY"}}}
    with pytest.raises(RuntimeError, match="ALL_SEMANTIC_REQUESTS_FAILED"):
        validate_request_outcomes({"a": {}}, {}, failures)
    validate_request_outcomes({"a": {}, "b": {}}, {"b": np.ones(2)}, failures)


def test_report_analysis_keeps_fixed_primary_negative_controls_and_null_deltas(tmp_path):
    from static_ovmap.cvpr_compact.tables import result_store
    from static_ovmap.cvpr_compact.reports import analyze_results, render_reports

    spec, scenes, pools, reference, diagnosis, timing = _table_inputs(tmp_path)
    store = result_store(spec, scenes, pools, tmp_path)
    result = analyze_results(spec, store, diagnosis, timing)
    assert result["primary_method"] == "CT_A3_ER" and result["deployment"] == "N0_UNCHANGED"
    delta = result["cohorts"]["replica8"]["A3_vs_A5_pp"]
    assert delta["apall"] == pytest.approx(-2.)
    assert "CT_A5_FC_ONLY" in result["cohorts"]["replica8"]["A3_dominated_by"]
    expected_methods = [row["id"] for row in spec["methods"] if "replica8" in row["cohorts"]]
    assert result["cohorts"]["replica8"]["metric_pareto_methods"] == expected_methods
    assert set(result["cohorts"]["replica8"]["pool_identities"]) == set(expected_methods)
    assert {"CT_H_U2", "CT_G3"} <= set(result["cohorts"]["replica8"]["A3_dominated_by"])
    assert result["cohorts"]["scannet_cf18"]["A3_vs_A4_pp"]["miou"] is None
    assert result["cohorts"]["replica8"]["guardrails_vs_A1"]["APall_positive"]
    assert result["cohorts"]["replica8"]["interaction_pp"]["apall"] == pytest.approx(0.)
    assert result["recovery"]["G3_vs_G1"]["mean_seconds_delta"] == 1.
    assert result["external_values_used_in_paired_analysis"] is False
    timing.update(model_loading_seconds_total=12.5,
                  hardware={"name": "synthetic GPU", "compute_capability": "8.6", "uuid": "synthetic-uuid"})
    binding = {"identity": "synthetic-binding", "output_root": str(tmp_path), "repository_root": str(tmp_path)}
    reports = render_reports(binding, store, diagnosis, timing, reference, result, "a" * 40)
    restored = json.loads(json.dumps(result, sort_keys=True))
    assert render_reports(binding, store, diagnosis, timing, reference, restored, "a" * 40) == reports
    assert set(reports) == {"COMPACT_TABLES_RESULTS.md", "COMPACT_TABLES_HANDOFF.md",
                           "COMPACT_TABLES_SELECTION.md", "COMPACT_TABLES_CLAIMS.md"}
    assert "-2.00" in reports["COMPACT_TABLES_RESULTS.md"]
    assert "12.50 seconds" in reports["COMPACT_TABLES_RESULTS.md"]
    assert "AUTHOR_PROTOCOL_AS_REPORTED" in reports["COMPACT_TABLES_CLAIMS.md"]
    from static_ovmap.cvpr_compact.costs import physical_ledger
    cost_data = {"status": "COMPLETE", "method_dependencies": [{"scene": "room0", "method_id": "CT_A5_FC_ONLY",
        "dependency_costs": {"status": "COMPLETE", "native_required_crop_inputs": 6, "FC_required_image_inputs": 2,
                             "FC_selected_mask_inputs": 3, "failed_request_count": 1}}],
                 "physical_payments": physical_ledger([])}
    cost_data["identity"] = canonical_digest(cost_data)
    reports = render_reports(binding, store, diagnosis, timing, reference, result, "a" * 40, cost_data=cost_data)
    assert "Native-painted support is a required A5 prerequisite" in reports["COMPACT_TABLES_RESULTS.md"]
    assert "CT_A5_FC_ONLY | -- | -- | -- | -- | 1/8" in reports["COMPACT_TABLES_RESULTS.md"]
    assert "tables/costs.json" in reports["COMPACT_TABLES_HANDOFF.md"]


def test_partial_reports_preserve_unavailable_comparisons_and_actual_negative_controls(tmp_path):
    from static_ovmap.cvpr_compact.tables import result_store
    from static_ovmap.cvpr_compact.reports import analyze_results, render_reports

    spec, scenes, pools, reference, diagnosis, timing = _partial_table_inputs(tmp_path)
    store = result_store(spec, scenes, pools, tmp_path)
    result = analyze_results(spec, store, diagnosis, timing)
    assert result["status"] == "PARTIAL_WITH_TECHNICAL_BLOCKS"
    assert result["primary_method"] == "CT_A3_ER" and result["deployment"] == "N0_UNCHANGED"
    replica = result["cohorts"]["replica8"]
    assert replica["A3_dominated_by"] is None
    assert all(value is None for value in replica["A3_vs_A1_pp"].values())
    assert all(value is None for value in replica["interaction_pp"].values())
    assert replica["guardrails_vs_A1"]["all_preferences_met"] is None
    assert replica["metric_pareto_scope_complete"] is False
    assert set(replica["metric_pareto_unavailable"]) == {"CT_A2_R", "CT_A3_ER", "CT_A5_FC_ONLY", "CT_G3"}
    assert replica["metric_pareto_frontier"] == ["CT_H_U2"]
    assert result["cohorts"]["scannet_cf18"]["A3_vs_A5_pp"]["apall"] == pytest.approx(-2.)
    for row in result["recovery"].values():
        assert row["source_available_delta"] is None and row["mean_seconds_delta"] is None
        assert row["matching_attribution_ambiguous"] is None
    timing.update(model_loading_seconds_total=7.5,
                  hardware={"name": "synthetic GPU", "compute_capability": "8.6", "uuid": "synthetic-uuid"})
    binding = {"identity": "synthetic-binding", "output_root": str(tmp_path), "repository_root": str(tmp_path)}
    reports = render_reports(binding, store, diagnosis, timing, reference, result, "a" * 40)
    text = reports["COMPACT_TABLES_RESULTS.md"]
    assert "168/172" in text and "10/14" in text and "22/24" in text
    assert "office1" in text and "-2.00" in text
    assert "None/" not in text and "Internal scientific coverage: COMPLETE" not in text
    assert "All eight scenes contribute to each cold mean" not in text
    assert "PARTIAL_WITH_TECHNICAL_BLOCKS" in reports["COMPACT_TABLES_CLAIMS.md"]


def test_actual_physical_collector_retains_independent_u2_and_skips_only_proven_unreserved_blocks(tmp_path):
    from static_ovmap.cvpr_compact.tables import collect_actual_physical_costs
    from static_ovmap.cvpr_compact.timing import record_blocked_measurement
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    _, projected, failed, _, scope = partial_scope_fixture()
    root = tmp_path / "task"
    historical = tmp_path / "parent"
    index = ConsumptionIndex()

    def save(path, value):
        value = copy.deepcopy(value)
        value["identity"] = canonical_digest({key: row for key, row in value.items() if key != "identity"})
        atomic_write_json(path, value)
        return index.identity(path)

    def worker(images, pools=0, **extra):
        return {"status": "COMPLETE", "scene": "office1", "physical_image_encodings": images,
                "physical_region_poolings": pools, "inputs": [], "outputs": [], **extra}

    native = historical / "native.json"
    fc = historical / "fc/receipt.json"
    mapping = historical / "map.json"
    save(native, worker(6))
    save(fc, worker(2, 3))
    save(mapping, worker(0))
    failed.update(physical_image_encodings=3, physical_region_poolings=0, inputs=[], outputs=[])
    failed_path = root / "recovery/office1/receipt.json"
    failed_item = save(failed_path, failed)
    save(root / "recovery/office1/receipt.failed_123.json", failed)
    projected_path = root / "projected_views/office1/manifest.json"
    projected_item = save(projected_path, projected)
    save(root / "recovery/office1_U2/receipt.json", worker(0, cold=False, GT_input=False, arms=["U2"]))
    save(root / "native_recovery/office1/receipt.json", worker(6))
    binding = {"identity": "synthetic-binding", "gpu": "2", "output_root": str(root), "path_map": [],
        "cohorts": {"replica8": ["office1"], "scannet_cf18": []},
        "scenes": {"office1": {"native_query_receipt": str(native), "fc_root": str(fc.parent),
                               "parent_map_receipt": str(mapping)}}}
    u2_call = root / "timing/office1/ARCHIVED_U2_FC/call/receipt.json"
    save(u2_call, worker(0, cold=True, arms=["U2"]))
    save(u2_call.parent.parent / "receipt.json", {"status": "COMPLETE", "scene": "office1",
        "arm": "ARCHIVED_U2_FC", "recovery_receipt": str(u2_call), "inputs": [], "outputs": []})
    from static_ovmap.cvpr_compact.partial_execution import semantic_block
    failed_document = json.loads(failed_path.read_text())
    projected_document = json.loads(projected_path.read_text())
    for arm in ("G1_FC", "G3_FC"):
        parent = {"status": "BLOCKED_TECHNICAL_PARENT", "scene": "office1", "arm": arm,
            "technical_block": semantic_block(failed_document, projected_document, arm[:2]),
            "failed_receipt": failed_item, "projected_manifest": projected_item}
        parent["identity"] = canonical_digest(parent)
        record_blocked_measurement(binding, "office1", arm, parent,
            root / "timing/office1" / arm / "receipt.json", index)
    result = collect_actual_physical_costs(binding, ["office1"], index=index)
    paths = [row["receipt"]["path"] for row in result["workers"]]
    assert str(root / "recovery/office1_U2/receipt.json") in paths
    assert str(root / "recovery/office1/receipt.failed_123.json") in paths
    assert len(paths) == len(set(paths)) == 8
    assert result["current_task_warm"]["FC_IMAGE"] == 6
    assert result["current_task_cold"]["FC_IMAGE"] == 0
    assert result["historical_parent"]["FC_IMAGE"] == 2
    assert result["complete_cold_call_count"] == 1 and result["unreserved_blocked_cold_leaf_count"] == 2
    save(root / "timing/office1/G1_FC/call/receipt.json", worker(1))
    with pytest.raises(ValueError, match="blocked|unreserved|call"):
        collect_actual_physical_costs(binding, ["office1"], index=index)


def test_cost_usage_unifies_legacy_and_new_fc_tensors_and_retains_failed_masks():
    from static_ovmap.cvpr_compact.costs import fc_usage, union_usage

    mask = "a" * 64
    legacy_key = canonical_digest({"image": "rgb-key", "mask": mask,
                                   "region": "original_signed_mask_pooling"})
    legacy = {"physical_model_identity": "model", "requests": {"old": {
        "status": "COMPLETE", "image_content_identity": "rgb-key", "content_identity": legacy_key}},
        "required_image_contents": {"rgb-key": 1}}
    manifest = {"requests": {"old": {"target_mask_sha256": mask}}}
    old = fc_usage(legacy, ["old"], "F", manifest=manifest,
                   dense_records={"rgb-key": {"input_tensor_key": "tensor", "image_content_key": "rgb-key"}})
    rows = {"alias": {"status": "COMPLETE", "image_content_key": "rgb-key",
                       "input_tensor_key": "tensor", "target_mask_sha256": mask},
            "failure": {"status": "UNAVAILABLE_TECHNICAL_FAILURE", "reason": "EMPTY_DENSE_MASK_SUPPORT",
                        "image_content_key": "rgb-key", "input_tensor_key": "tensor", "target_mask_sha256": "b" * 64}}
    current = fc_usage({"physical_model_identity": "model", "requests": rows}, rows, "G1_FC")
    combined = union_usage(old, current)
    assert combined["FC_required_image_inputs"] == 1
    assert combined["FC_selected_mask_inputs"] == 2
    assert combined["logical_attempted_requests"] == 3
    assert combined["failed_request_count"] == 1
    assert combined["status"] == "COMPLETE"
    assert combined["requests"]["G1_FC:failure"]["reason"] == "EMPTY_DENSE_MASK_SUPPORT"
    missing = fc_usage({"physical_model_identity": "model", "requests": {
        "failure": {"status": "UNAVAILABLE_TECHNICAL_FAILURE", "reason": "DECODE_FAILED"}}},
        ["failure"], "G1_FC")
    assert missing["status"] == "PARTIAL_CONTENT_IDENTITIES"
    assert missing["required_image_encodings"] is None
    assert missing["failed_request_count"] == 1


def test_method_costs_keep_native_support_prerequisite_and_union_only_required_sources():
    from static_ovmap.cvpr_compact.costs import native_usage, fc_usage, method_costs

    support = native_usage({"n": 6}, [{"request_id": "n", "status": "COMPLETE"}], "N_SUPPORT")
    query = native_usage({"n": 6, "q": 6}, [{"request_id": "q", "status": "COMPLETE"}], "Q")
    def fc(stage, rows):
        return fc_usage({"physical_model_identity": "fc-model", "requests": {
            rid: {"status": "COMPLETE", "input_tensor_key": tensor, "target_mask_sha256": mask}
            for rid, tensor, mask in rows}}, [row[0] for row in rows], stage)
    F = fc("F", [("f", "shared", "a" * 64)])
    recovery = fc("G1_FC", [("r1", "shared", "b" * 64), ("r2", "new", "c" * 64)])
    inventory = {"status": "COMPLETE", "support": support, "query": query, "F": F,
                 "scene": "fixture", "common_map_receipt": {"sha256": "d" * 64}}
    inventory["identity"] = canonical_digest(inventory)
    spec = json.loads(SPEC.read_text())
    values = {}
    for method in spec["methods"][:6]:
        selected = native_usage({"n": 6, "r": 6}, [{"request_id": "r", "status": "COMPLETE"}], "G1_NATIVE")
        if method["recovery"] == "G1_FC":
            selected = recovery
        elif method["recovery"] == "NONE":
            selected = None
        values[method["id"]] = method_costs(method, inventory, selected)
    assert [values[name]["required_image_encodings"] for name in
            ("CT_A0_NATIVE", "CT_A1_E", "CT_A2_R", "CT_A3_ER", "CT_A4_NATIVE_RECOVERY", "CT_A5_FC_ONLY")] == [6, 13, 8, 14, 19, 8]
    a5 = values["CT_A5_FC_ONLY"]
    assert a5["native_support_prerequisite_included"]
    assert a5["required_components"] == ["N_SUPPORT", "F", "G1_FC"]
    assert a5["FC_required_image_inputs"] == 2 and a5["FC_selected_mask_inputs"] == 3
    assert a5["query_attempts"] == 0 and values["CT_A3_ER"]["query_attempts"] == 1
    assert a5["standalone_wall_seconds"] is None
    assert a5["timing_basis"] == "LOGICAL_CONTENT_UNION_NOT_A_COLD_WALL_TIME"
    with pytest.raises(ValueError, match="crop|content"):
        from static_ovmap.cvpr_compact.costs import union_usage
        union_usage(support, native_usage({"n": 5}, [], "conflict"))


def test_shared_physical_costs_preserve_paid_failures_without_multiplying_method_totals():
    from static_ovmap.cvpr_compact.costs import physical_stage

    value = {"status": "FAILED", "physical_image_encodings": 2,
             "physical_region_poolings": 1, "encoder_batch_calls": 2,
             "elapsed_seconds": 3., "model_load_seconds": .5,
             "requests": {"paid": {"status": "UNAVAILABLE_TECHNICAL_FAILURE", "reason": "CUDA_OUT_OF_MEMORY"}}}
    row = physical_stage("FC_RECOVERY_SHARED", value, {"sha256": "a" * 64}, current_task=True)
    assert row["status"] == "FAILED"
    assert row["current_task_physical_image_inputs"] == 2
    assert row["current_task_physical_region_poolings"] == 1
    assert row["failed_request_count"] == 1
    assert row["method_costs_may_be_summed"] is False
    historical = physical_stage("N_Q_SHARED", value, {"sha256": "a" * 64}, current_task=False)
    assert historical["historical_physical_image_inputs"] == 2
    assert historical["current_task_physical_image_inputs"] == 0


def test_physical_ledger_counts_shared_workers_once_and_separates_cold_and_historical_work():
    from static_ovmap.cvpr_compact.costs import physical_ledger

    receipt = {"status": "COMPLETE", "physical_image_encodings": 1, "physical_region_poolings": 2,
               "elapsed_seconds": .5, "requests": {}}
    def record(stage, path, *, current=True, cold=False, value=None):
        return {"scene": "fixture", "stage": stage, "receipt": value or receipt,
                "file": {"path": path, "sha256": "a" * 64}, "current_task": current, "cold": cold}
    warm = record("FC_RECOVERY_SHARED", "/fixture/shared.json")
    cold = record("FC_RECOVERY_COLD", "/fixture/cold.json", cold=True)
    parent = record("FC_STATIC_SHARED", "/parent/static.json", current=False,
                    value={**receipt, "physical_image_encodings": 8})
    ledger = physical_ledger([warm, warm, cold, parent])
    assert len(ledger["workers"]) == 3
    assert ledger["current_task_warm"]["FC_IMAGE"] == 1
    assert ledger["current_task_cold"]["FC_IMAGE"] == 1
    assert ledger["historical_parent"]["FC_IMAGE"] == 8
    assert ledger["current_task_warm_FC_region_poolings"] == 2
    assert ledger["current_task_cold_FC_region_poolings"] == 2
    assert ledger["logical_method_budgets_may_be_summed"] is False
    unknown = record("FC_RECOVERY_SHARED", "/fixture/unknown.json",
                     value={"status": "FAILED", "elapsed_seconds": .1, "requests": {}})
    ledger = physical_ledger([warm, unknown])
    assert ledger["current_task_warm"]["FC_IMAGE"] is None
    assert ledger["current_task_warm_known_lower_bound"]["FC_IMAGE"] == 1


def test_base_cost_inventory_recovers_paid_failed_dense_inputs_from_original_ledger(tmp_path):
    from static_ovmap.cvpr_compact.costs import load_base_cost_inventory
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex
    from static_ovmap.module_validation.boundary_jobs import file_identity
    from static_ovmap.module_validation.contracts import atomic_write_json

    def write(name, value):
        path = tmp_path / name
        atomic_write_json(path, value)
        return path
    dense_records = {}
    for image in ("good", "failed"):
        tensor = canonical_digest(image)
        path = tmp_path / "cache/model/dense" / (tensor + ".npz")
        _write_npz(path, {"dense": np.zeros((1, 2, 2, 2), np.float32)})
        record = {"input_tensor_key": tensor, "image_content_key": image, "arrays": file_identity(path)}
        atomic_write_json(path.with_suffix(".json"), record)
        dense_records[image] = record
    native = write("native.json", {"record_key": "native-key"})
    mapping = write("map.json", {"status": "COMPLETE", "scene": "fixture", "map_id": "BB00_NATIVE"})
    requests = {image: {"target_mask_sha256": canonical_digest(image + "mask")} for image in dense_records}
    manifest = {"native_record_key": "native-key", "scene_id": "fixture", "requests": requests}
    manifest["identity"] = canonical_digest(manifest)
    write("fc_requests.json", manifest)
    logical = {"attempts": 0, "crop_inputs": 0, "failures": 0, "successes": 0}
    decisions = write("query.json", {"frames": [], "logical": logical})
    nq = write("native_query.json", {"status": "COMPLETE", "map_id": "BB00_NATIVE",
        "native_required_content": {"native": 6}, "standalone_required_content": {"native": 6},
        "native_requests": [{"request_id": "native", "status": "COMPLETE"}],
        "query_logical_ledger": logical, "query_decisions": file_identity(decisions)})
    rows = {image: {"status": "COMPLETE" if image == "good" else "UNAVAILABLE_TECHNICAL_FAILURE",
        "error": "EMPTY_DENSE_MASK_SUPPORT" if image == "failed" else None,
        "image_content_identity": image, "content_identity": canonical_digest({"image": image,
            "mask": requests[image]["target_mask_sha256"], "region": "original_signed_mask_pooling"})}
        for image in requests}
    write("fc/receipt.json", {"status": "COMPLETE", "physical_model_identity": "model", "requests": rows,
        "request_manifest": manifest["identity"], "required_image_contents": {image: 1 for image in requests},
        "required_dense_receipts": {"good": str(Path(dense_records["good"]["arrays"]["path"]).with_suffix(".json"))},
        "inputs": [row["arrays"] for row in dense_records.values()]})
    binding = {"path_map": {}, "inputs": []}
    data = {"native_query_receipt": str(nq), "fc_root": str(tmp_path / "fc"),
            "FC_physical_model_identity": "model", "predictions": {"NATIVE_READOUT": str(native)},
            "parent_map_receipt": str(mapping)}
    result = load_base_cost_inventory(binding, data, index=ConsumptionIndex())
    assert result["F"]["status"] == "COMPLETE"
    assert result["F"]["FC_required_image_inputs"] == 2
    assert result["F"]["FC_selected_mask_inputs"] == 2 and result["F"]["failed_request_count"] == 1
    assert result["F"]["required_image_encodings"] == 2
    bad = dense_records["failed"]
    bad["arrays"]["sha256"] = "0" * 64
    atomic_write_json(Path(bad["arrays"]["path"]).with_suffix(".json"), bad)
    with pytest.raises(ValueError, match="dense|array|input|sha"):
        load_base_cost_inventory(binding, data, index=ConsumptionIndex())


def test_execution_leaf_has_three_attempt_budget_and_never_replays_success(tmp_path):
    import os
    import sys
    from static_ovmap.cvpr_compact.runtime import execute_leaf

    argv = [sys.executable, "-c", "raise SystemExit(7)"]
    log = tmp_path / "failed.log"
    identity = canonical_digest(argv)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="exited 7"):
            execute_leaf(argv, tmp_path, log, env=os.environ, input_identity=identity)
    with pytest.raises(RuntimeError, match="exhausted"):
        execute_leaf(argv, tmp_path, log, env=os.environ, input_identity=identity)
    assert len(json.loads((tmp_path / ("command_" + identity + ".json")).read_text())["attempts"]) == 3
    assert len(list(tmp_path.glob("failed.attempt_*.log"))) == 3
    success = [sys.executable, "-c", "print('ok')"]
    identity = canonical_digest(success)
    row = execute_leaf(success, tmp_path, tmp_path / "success.log", env=os.environ, input_identity=identity)
    assert row["status"] == "COMPLETE" and row["controller_start_ticks"] and row["child_start_ticks"]
    with pytest.raises(RuntimeError, match="never execute again"):
        execute_leaf(success, tmp_path, tmp_path / "success.log", env=os.environ, input_identity=identity)


def test_orphan_retains_command_lock_and_cannot_be_replaced(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    from static_ovmap.cvpr_compact.runtime import LiveLeafError, execute_leaf, process_state

    argv = [sys.executable, "-c", "import time; time.sleep(15)"]
    log = tmp_path / "orphan.log"
    identity = canonical_digest(argv)
    code = ("import json,os; from static_ovmap.cvpr_compact.runtime import execute_leaf; "
            "execute_leaf(json.loads(os.environ['TEST_LEAF_ARGV']), os.environ['TEST_LEAF_CWD'], "
            "os.environ['TEST_LEAF_LOG'], env=os.environ, input_identity=os.environ['TEST_LEAF_ID'])")
    environment = {**os.environ, "TEST_LEAF_ARGV": json.dumps(argv), "TEST_LEAF_CWD": str(tmp_path),
                   "TEST_LEAF_LOG": str(log), "TEST_LEAF_ID": identity}
    parent = subprocess.Popen([sys.executable, "-c", code], env=environment)
    child = None
    try:
        deadline = time.monotonic() + 5
        ledger = tmp_path / ("command_" + identity + ".json")
        while time.monotonic() < deadline:
            if ledger.exists():
                row = json.loads(ledger.read_text())["attempts"][-1]
                child = row.get("child_pid")
                if child:
                    break
            time.sleep(.01)
        assert child and process_state(child)["live"]
        parent.kill()
        parent.wait(timeout=5)
        prepared = []
        with pytest.raises(LiveLeafError, match="lock is held"):
            execute_leaf(argv, tmp_path, log, env=environment, input_identity=canonical_digest("changed-source"),
                         prepare=lambda: prepared.append(True))
        assert prepared == [] and len(json.loads(ledger.read_text())["attempts"]) == 1
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child and process_state(child):
            os.kill(child, signal.SIGTERM)


def test_distinct_evaluation_commands_may_run_concurrently(tmp_path):
    import os
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from static_ovmap.cvpr_compact.runtime import execute_leaf

    def run(scene):
        argv = [sys.executable, "-c", "import time; time.sleep(.2)", scene]
        return execute_leaf(argv, tmp_path, tmp_path / (scene + ".log"), env=os.environ, input_identity=canonical_digest(argv))
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, ("office0", "office1", "room0")))
    assert all(row["status"] == "COMPLETE" for row in rows)
    assert max(row["started_at_unix"] for row in rows) - min(row["started_at_unix"] for row in rows) < .2


def test_final_freeze_validation_rejects_stale_source_and_failed_smoke(tmp_path):
    from static_ovmap.cvpr_compact.freezing import verified_validation, main_outcomes_exist
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    binding = {"output_root": str(tmp_path), "spec": str(SPEC)}
    inventory = [{"path": "current.py", "sha256": "a" * 64, "bytes": 10}]
    proof = {"status": "PASS", "stage": "FINAL_PRODUCTION_VALIDATION", "exit_code": 0, "tests_passed": 1,
             "argv": ["python", "-m", "pytest", "-q", "tests/evaluation/test_cvpr_compact_contracts.py"],
             "implementation_sources": inventory}
    proof["identity"] = canonical_digest(proof)
    atomic_write_json(tmp_path / "validation/final_contract_tests.json", proof)
    assert verified_validation(binding, "final_contract_tests", inventory, ConsumptionIndex())["sha256"]
    with pytest.raises(ValueError, match="stale"):
        verified_validation(binding, "final_contract_tests", [], ConsumptionIndex())
    proof = {"status": "COMPLETE", "scene": "scene0056_00", "logical_condition_count": 8,
             "new_development_map_count": 1, "main_performance_outcomes_seen": False,
             "implementation_sources": inventory}
    proof["identity"] = canonical_digest(proof)
    atomic_write_json(tmp_path / "validation/final_smoke.json", proof)
    with pytest.raises(ValueError, match="existing"):
        verified_validation(binding, "final_smoke", inventory, ConsumptionIndex())
    assert not main_outcomes_exist(binding)
    atomic_write_json(tmp_path / "readouts/scene0011_00/BB00_NATIVE/receipt.json", {"status": "RUNNING"})
    assert main_outcomes_exist(binding)


def test_publication_catalog_covers_package_and_push_requires_full_named_sha():
    from static_ovmap.cvpr_compact.publication import check_public_json, requirement_catalog, verify_remote_sha

    catalog = requirement_catalog()
    assert len({row["id"] for row in catalog["requirements"]}) == len(catalog["requirements"])
    for name in ("CODEX_FINAL_EXECUTION_EN.md", "IMPLEMENTATION_CONTRACTS.md", "TABLE_CONTRACTS.md", "SOURCE_EVIDENCE.md",
                 "PROTOCOL_SPEC.json", "TABLE_BINDINGS.json"):
        assert any(row["source"] == name for row in catalog["requirements"])
    sha, branch = "a" * 40, "research/ovimap-cvpr-compact-tables-v1"
    assert verify_remote_sha(sha, sha + "\trefs/heads/" + branch + "\n", branch) == sha
    for local, output in ((sha[:7], sha + "\trefs/heads/" + branch), (sha, "b" * 40 + "\trefs/heads/" + branch),
                          (sha, sha + "\trefs/heads/main")):
        with pytest.raises(ValueError, match="full40"):
            verify_remote_sha(local, output, branch)
    check_public_json({"scores": [0., 1.], "model_identity": "frozen"})
    with pytest.raises(ValueError, match="secret"):
        check_public_json({"data": {"access_token": "secret"}})


@pytest.mark.parametrize("partial", [False, True])
def test_release_contains_real_scores_decisions_matching_and_deterministic_compression(tmp_path, monkeypatch, partial):
    import gzip
    from static_ovmap.cvpr_compact import publication
    from static_ovmap.cvpr_compact.protocol import experiment_matrix, load_spec
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    spec = load_spec(SPEC)
    root, repo = tmp_path / "run", tmp_path / "repo"
    binding = {"output_root": str(root), "repository_root": str(repo), "spec": str(SPEC),
               "cohorts": spec["cohorts"], "inputs": [], "path_map": {}}
    matrix = experiment_matrix(spec)
    coverage = {"identity": "coverage", "freeze_revision": "a" * 40}
    def audit(binding, *, index):
        atomic_write_json(root / "publication/scientific_coverage.json", coverage)
        return coverage
    monkeypatch.setattr(publication, "audit_measurements", audit)
    def write(relative, value):
        path = root / relative
        atomic_write_json(path, value)
        return path
    for name in ("matrix.json", "resolved_inputs.json", "exposure_ledger.json", "tables/result_store.json",
                 "reports/analysis.json", "external/reference.json", "validation/final_contract_tests.json",
                 "validation/final_smoke.json", "validation/final_table_visual_qa.json", "diagnostics/replica8_recovery.json",
                 "diagnostics/receipt.json", "timing/pool.json"):
        write(name, {"value": "fixture"})
    write("tables/costs.json", {"physical_payments": {"workers": []}})
    original_scores = {"scores": [.25, -.75], "padding": "x" * 70000}
    _, _, _, _, partial_scope = partial_scope_fixture()
    for anchor in matrix["anchors"]:
        scene = anchor["scene"]
        query_decisions = write("original/" + scene + "/query_decisions.json", {"selected": ["actual-q-request"]})
        nq = write("original/" + scene + "/native_query_receipt.json",
                   {"query_decisions": ConsumptionIndex().identity(query_decisions)})
        sources = {name: {"path": str(write("original/" + scene + "/" + name + ".json", original_scores))}
                   for name in ("N", "Q", "F")}
        write("contexts/" + scene + ".json", {"sources": sources, "native_query_receipt": str(nq)})
        recovery = write("original/" + scene + "/G1.json", {"scores": [.3, .6], "label": 2})
        native = write("original/" + scene + "/G1_NATIVE.json", {"scores": [.6, .3], "label": 1})
        projected = write("original/" + scene + "/views.json", {"g1": ["selected-mask"]})
        for directory in ("projected_views", "recovery", "native_recovery", "predictions"):
            write(directory + "/" + scene + "/receipt.json", {"manifest": str(projected),
                  "source": {"path": str(native)}, "inputs": [], "outputs": []})
        blocked_scene = partial and scene == "office1"
        if blocked_scene:
            write("recovery/office1/receipt.json", {"status": "FAILED", "inputs": [], "outputs": []})
            write("predictions/office1/receipt.json", {**partial_scope, "status": "PARTIAL_PREDICTIONS_LOCKED",
                                                      "inputs": [], "outputs": []})
            write("recovery/office1_U2/receipt.json", {"sources": {"U2": {"path": str(recovery)}},
                                                      "inputs": [], "outputs": []})
        else:
            write("recovery/" + scene + "/regions/receipt.json", {"sources": {"G1": {"path": str(recovery)}}})
        evaluated, registry = [], {}
        for planned in [row for row in matrix["outputs"] if row["scene"] == scene]:
            method = planned["method"]
            if blocked_scene and method in partial_scope["blocked_methods"]:
                continue
            manifest = {"prediction_key": method, "record_key": method + "-record", "instance_ranks": [[7, .5]],
                        "metadata": {"unknown_incumbent_semantics_preserve_support": True,
                                     "owner_semantic_decisions": {"7": 1, "8": 0, "9": 2}},
                        "owner_ids": {"sha256": "owner"}, "semantic_labels": {"sha256": "label"}}
            write("predictions/" + scene + "/" + method + "/manifest.json", manifest)
            scoring = write("scoring/" + scene + "/" + method + "/scoring.json",
                            {"view": {"7": {"label": 1, "rank": .5}},
                             "manifest": str(root / "scoring" / scene / method / "predictions.txt")})
            for name in ("matches.json.gz", "trace.json.gz"):
                scoring.with_name(name).write_bytes(gzip.compress(b'{"actual_score_entry":7}', mtime=0))
            evaluated.append({"method": method, "evaluation_receipt": str(scoring)})
            registry[method] = {"unknown_semantic_positive_owners": [8]}
        write("evaluation/" + scene + "/receipt.json", {"rows": evaluated, "registry_checks": registry})
        write("predictions/" + scene + "/existing_D2_probability_audit.json", {"original_probabilities": [.25, .75]})
        write("diagnostics/" + scene + ".json", {"candidate_N": 3})
        if anchor["cohort"] == "scannet_cf18":
            for path in ("prepare/" + scene + "/receipt.json", "frontend/" + scene + "/cropformer/receipt.json",
                         "maps/" + scene + "/BB00_NATIVE/map_receipt.json", "readouts/" + scene + "/BB00_NATIVE/receipt.json",
                         "readouts/" + scene + "/BB00_NATIVE/fc/receipt.json"):
                write(path, {"status": "COMPLETE"})
    for planned in matrix["pools"]:
        write("pools/" + planned["cohort"] + "/" + planned["method"] + ".json", {"ordered_scene_ids": planned["scene_order"]})
    for planned in matrix["timings"]:
        leaf = "timing/" + planned["scene"] + "/" + planned["arm"]
        if partial and planned["scene"] == "office1" and planned["arm"] in ("G1_FC", "G3_FC"):
            write(leaf + "/receipt.json", {"status": "BLOCKED_UNMEASURED", "scene": "office1", "arm": planned["arm"],
                "technical_block": partial_scope["blocked_sources"][planned["arm"]], "measured_seconds": None,
                "parity": None, "physical_calls_reserved": 0, "measurement_reserved": False, "cold_call_executed": False,
                "inputs": [], "outputs": []})
        else:
            call = write(leaf + "/call/receipt.json", {"actual_seconds": .5})
            write(leaf + "/receipt.json", {"recovery_receipt": str(call), "inputs": [], "outputs": []})
    legacy = repo / publication.RELEASE_ROOT / "sources/office0/N.json.gz"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(gzip.compress((root / "original/office0/N.json").read_bytes(), mtime=0))
    legacy_identity = ConsumptionIndex().identity(legacy)
    first = publication.build_release(binding)
    second = publication.build_release(binding)
    assert first == second
    release = repo / publication.RELEASE_ROOT
    archive = first["evidence_archive"]
    extracted = tmp_path / "extracted"
    restored = publication.verify_evidence_archive(repo / archive["file"]["path"], archive["members"], extract_to=extracted)
    assert restored["file_count"] == len(archive["members"])
    assert json.loads((extracted / "sources/office0/N.json").read_text()) == original_scores
    source_path = root / "original/office0/N.json"
    assert (extracted / "sources/office0/N.json").read_bytes() == source_path.read_bytes()
    decision_paths = list((extracted / "decisions").glob("*/*.json"))
    assert len(decision_paths) == (168 if partial else 172)
    assert json.loads(decision_paths[0].read_text())["registry_checks"]["unknown_semantic_positive_owners"] == [8]
    assert json.loads(decision_paths[0].read_text())["all_positive_owner_labels"] == {"7": 1, "8": 0, "9": 2}
    assert len(list((extracted / "evaluation").glob("*/*/matches.json.gz"))) == (168 if partial else 172)
    assert len(list((extracted / "timing").glob("*/*/receipt.json"))) == 24
    assert len(list((extracted / "timing").glob("*/*/call/receipt.json"))) == (22 if partial else 24)
    if partial:
        assert (extracted / "recovery/office1_U2/receipt.json").is_file()
        assert not (extracted / "recovery/office1/regions/receipt.json").exists()
    assert (release / "tables/result_store.json").is_file()
    assert not (release / "sources/office0/N.json.gz").exists()
    saved_legacy = root / "publication/previous_release_files" / legacy_identity["sha256"]
    assert ConsumptionIndex().identity(saved_legacy)["sha256"] == legacy_identity["sha256"]
    invalid = copy.deepcopy(archive["members"])
    invalid[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        publication.verify_evidence_archive(repo / archive["file"]["path"], invalid)
    invalid[0]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path"):
        publication.verify_evidence_archive(repo / archive["file"]["path"], invalid)
    assert first["bytes_excluding_bundle_and_primary_review"] < publication.LIMIT_BYTES
    monkeypatch.setattr(publication, "LIMIT_BYTES", 1)
    with pytest.raises(RuntimeError, match="50MiB"):
        publication.build_release(binding)


def test_primary_review_allows_only_explicit_actual_blocks_without_claiming_completion(tmp_path, monkeypatch):
    from static_ovmap.cvpr_compact import publication
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex

    root, repo = tmp_path / "run", tmp_path / "repo"
    index = ConsumptionIndex()
    catalog = {"identity": "synthetic-catalog", "requirements": [{"id": "implementation"}, {"id": "full_cold_mean"}]}
    monkeypatch.setattr(publication, "requirement_catalog", lambda: catalog)
    monkeypatch.setattr(publication, "implementation_inventory", lambda *args, **kwargs: [])
    evidence = root / "original_block.json"
    atomic_write_json(evidence, {"synthetic_original_failure": True})
    coverage = {"blocked_required_dependencies": ["timing/office1/G1_FC"]}
    coverage["identity"] = canonical_digest(coverage)
    atomic_write_json(root / "publication/scientific_coverage.json", coverage)
    files = []
    for name in publication.REPORT_NAMES:
        path = repo / "docs/paper/static_ovmap" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic report for review-status regression only.\n")
        files.append(index.identity(path))
    review = {"status": "PASS_WITH_TECHNICAL_BLOCKS", "reviewer": "PRIMARY_CODEX", "objective_complete": False,
        "requirement_catalog_identity": catalog["identity"], "bundle_identity": "synthetic-bundle",
        "unresolved_required_items": ["full_cold_mean"], "reviewed_files": files,
        "requirements": [{"requirement_id": "implementation", "status": "PROVEN_COMPLETE",
            "evidence": [index.identity(evidence)], "finding": "Synthetic complete contract."},
            {"requirement_id": "full_cold_mean", "status": "BLOCKED_TECHNICAL",
             "blocked_dependencies": ["timing/office1/G1_FC"], "evidence": [index.identity(evidence)],
             "finding": "The fixed eight-scene mean has one actual unmeasured block."}]}
    path = root / "validation/primary_review.json"

    def save(value):
        value["identity"] = canonical_digest({key: row for key, row in value.items() if key != "identity"})
        atomic_write_json(path, value)

    save(review)
    binding = {"output_root": str(root), "repository_root": str(repo)}
    assert publication.verify_primary_review(binding, {"identity": "synthetic-bundle", "files": []}) == review
    for change in ("claim_complete", "global_pass", "hide_unresolved", "invent_block", "missing_requirement"):
        changed = copy.deepcopy(review)
        if change == "claim_complete":
            changed["objective_complete"] = True
        elif change == "global_pass":
            changed["status"] = "PASS"
        elif change == "hide_unresolved":
            changed["unresolved_required_items"] = []
        elif change == "invent_block":
            changed["requirements"][1]["blocked_dependencies"] = ["timing/room0/G1_FC"]
        else:
            changed["requirements"].pop(0)
        save(changed)
        with pytest.raises(ValueError):
            publication.verify_primary_review(binding, {"identity": "synthetic-bundle", "files": []})
    save(review)
    members = [{"path": "scores/scene.json", "sha256": "a" * 64, "bytes": 10}]
    bundle = {"identity": "synthetic-bundle", "files": [], "evidence_archive": {"members": members}}
    with pytest.raises(ValueError, match="archiv"):
        publication.verify_primary_review(binding, bundle)
    review["archived_members_reviewed"] = members
    save(review)
    assert publication.verify_primary_review(binding, bundle) == review


def test_legacy_region_cache_restores_only_losslessly_widened_original_fp32(tmp_path):
    from static_ovmap.cvpr_compact.region_worker import FCSession
    from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex
    from static_ovmap.module_validation.contracts import atomic_write_json

    index = ConsumptionIndex()
    dense = tmp_path / "parent/model/dense/tensor.npz"
    _write_npz(dense, {"dense": np.ones((1, 3, 2, 2), np.float32)})
    atomic_write_json(dense.with_suffix(".json"), {"input_tensor_key": "tensor", "image_content_key": "image",
                      "arrays": index.identity(dense)})
    original = np.asarray([.1, .2, .3], np.float32)
    region = tmp_path / "parent/model/regions/old.npz"
    _write_npz(region, {"feature": original.astype(np.float64)})
    def receipt():
        atomic_write_json(region.with_suffix(".json"), {"content_identity": "old", "arrays": index.identity(region),
                          "dense_receipt": str(dense.with_suffix(".json"))})
    receipt()
    before = index.identity(region)
    session = FCSession.__new__(FCSession)
    session.index, session.text = index, np.zeros((2, 3), np.float32)
    session.cache = ContentCache([tmp_path / "parent"], tmp_path / "new", "model", index=index)
    vector, row, path, legacy = session._region_hit("new", "old", "tensor", "image")
    assert legacy and vector.dtype == np.float32
    np.testing.assert_array_equal(vector, original)
    assert row["cache_storage_dtype"] == "<f8"
    assert row["precision_restoration"] == "LOSSLESS_FP64_STORAGE_OF_ORIGINAL_FP32"
    assert index.identity(region) == before and not (tmp_path / "new").exists()
    _write_npz(region, {"feature": np.asarray([1/3, .2, .3], np.float64)})
    receipt()
    with pytest.raises(ValueError, match="lossless"):
        session._region_hit("new", "old", "tensor", "image")
