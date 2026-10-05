"""Production checks for exact rays, full-scene occlusion and bounded cold work."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from static_ovmap.cvpr_compact.projected_views import FullSceneProjector, camera_rays


def kernels():
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/kernels.py"
    assert path.is_file(), "runtime parity kernels are not implemented"
    return importlib.import_module("static_ovmap.runtime_parity.kernels")


def fixture_mesh(*, mixed=False):
    xyz = np.array([[-3,-3,2], [3,-3,2], [3,3,2], [-3,3,2],
                    [0,-3,1], [3,-3,1], [3,3,1], [0,3,1]], np.float32)
    faces = np.array([[0,1,2], [0,2,3], [4,5,6], [4,6,7]], np.int64)
    raw = np.array([9,9,9,9,0,0,0,0], np.int64)
    if mixed:
        raw[4:] = [2,3,2,3]
    K = np.array([[10,0,12], [0,10,10], [0,0,1]], np.float64)
    return xyz, faces, raw, K, np.eye(4), np.full((21,25), 2, np.float32)


def test_prepared_camera_preserves_original_fp32_ray_bytes():
    mod = kernels()
    K = np.array([[511.7,.03,14.2], [0,510.4,8.7], [0,0,1]], np.float64)
    pose = np.eye(4)
    pose[0,0] += 1e-7
    pose[:3,3] = [.123456789, -.991234567, .321012345]
    shape = (19,29)
    selected = np.array([0,1,28,29,145,289,549,550], np.int64)
    prepared = mod.PreparedCamera(K, pose, shape)
    assert prepared.rays(selected).tobytes() == camera_rays(K,pose,shape,0,np.prod(shape))[selected].tobytes()
    with pytest.raises(ValueError):
        prepared.rays(np.array([-1]))


@pytest.mark.parametrize("roi", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_optimized_projector_keeps_noncandidate_and_mixed_owner_occluders(roi, mixed):
    mod = kernels()
    xyz, faces, raw, K, pose, depth = fixture_mesh(mixed=mixed)
    reference = FullSceneProjector(xyz,faces,raw,batch_pixels=37).project_frame(K,pose,depth,[9])
    selected = mod.ExactProjector(xyz,faces,raw,roi=roi,batch_pixels=37).project_frame(K,pose,depth,[9])
    assert selected["counts"] == reference["counts"]
    assert reference["counts"][9]["depth_consistent_pixels"] < depth.size
    assert np.array_equal(selected["mask_for_owner"](9),reference["masks"][9])
    if roi:
        assert selected["diagnostic_scope"] == "queried_pixels"
        assert selected["total_first_hits"] is None


def test_roi_near_plane_and_camera_inside_use_full_frame():
    mod = kernels()
    K = np.array([[10,0,5], [0,10,5], [0,0,1]], np.float64)
    prepared = mod.PreparedCamera(K,np.eye(4),(11,11))
    bounds = np.array([[-1,-1,-1],[1,1,1]],np.float32)
    pixels, reason = prepared.query_union([bounds])
    assert np.array_equal(pixels,np.arange(121))
    assert reason == "UNCERTAIN_FULL_FRAME"


def test_roi_offscreen_empty_and_outward_image_edge_guard():
    mod = kernels()
    prepared = mod.PreparedCamera(np.eye(3),np.eye(4),(10,10))
    empty, reason = prepared.query_union([np.array([[100,100,2],[101,101,3]],np.float32)])
    assert not len(empty) and reason == "PROVEN_EMPTY"
    pixels, reason = prepared.query_union([np.array([[18,0,2],[19,1,2]],np.float32)])
    assert reason == "CONSERVATIVE_UNION" and 9 in pixels


def test_grouped_counts_handle_large_noncontiguous_owner_ids():
    mod = kernels()
    image = np.array([[0,10**12,0],[7,10**12,7]],np.int64)
    stats = mod.grouped_stats(image,[7,10**12,99])
    assert stats[7] == (2,[0,1,3,2])
    assert stats[10**12] == (2,[1,0,2,2])
    assert stats[99] == (0,None)


def test_lazy_top3_discards_only_strictly_worse_leading_terms():
    mod = kernels()
    def entry(area, frame, digest):
        return (SimpleNamespace(visible_pixels=area,frame_id=frame,target_mask_sha256=digest),None,None)
    values = [entry(200,1,"a"),entry(150,2,"b"),entry(100,3,"c")]
    def forbidden():
        pytest.fail("strictly worse candidate unnecessarily materialized")
    assert not mod.retain_top3(values,99,0,forbidden)
    calls = []
    def tied():
        calls.append(True)
        return entry(100,3,"b")
    assert mod.retain_top3(values,100,3,tied)
    assert calls == [True] and values[-1][0].target_mask_sha256 == "b"


def test_sparse_depth_threshold_matches_reference():
    mod = kernels()
    xyz, faces, raw, K, pose, depth = fixture_mesh()
    depth[:,0] = np.nan
    depth[:,1] = 2.04
    depth[:,2] = np.nextafter(np.float32(2.04),np.float32(3))
    reference = FullSceneProjector(xyz,faces,raw).project_frame(K,pose,depth,[9])
    fast = mod.ExactProjector(xyz,faces,raw,roi=True).project_frame(K,pose,depth,[9])
    assert fast["counts"] == reference["counts"]
    assert np.array_equal(fast["mask_for_owner"](9),reference["masks"][9])


def test_bound_producer_alias_must_match_measured_hash(tmp_path):
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/binding.py"
    assert path.is_file(), "runtime parity binding is not implemented"
    mod = importlib.import_module("static_ovmap.runtime_parity.binding")
    source, copied = tmp_path/"source.py", tmp_path/"copy.py"
    source.write_text("actual_operator = 1\n")
    copied.write_text("approximate_operator = 2\n")
    expected = mod.file_identity(source)
    with pytest.raises(ValueError,match="measured producer"):
        mod.validate_producer_alias(source,copied,expected)
    copied.write_bytes(source.read_bytes())
    assert mod.validate_producer_alias(source,copied,expected)["sha256"] == expected["sha256"]


def test_exclusive_stage_accounting_does_not_double_count_nested_spans():
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/instrumentation.py"
    assert path.is_file(), "exclusive instrumentation is not implemented"
    mod = importlib.import_module("static_ovmap.runtime_parity.instrumentation")
    ticks = iter([0.,1.,4.,5.])
    stages = mod.Stages(clock=lambda:next(ticks))
    with stages.span("registry_geometry"):
        with stages.span("view_search"):
            pass
    values = stages.finish(6.)
    assert values["registry_geometry"] == 2.
    assert values["view_search"] == 3.
    assert values["other_sync"] == 1.
    assert sum(values.values()) == 6.


def test_scientific_key_excludes_producer_but_retains_frame_mask_and_vocabulary():
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/views.py"
    assert path.is_file(), "scientific request mapping is not implemented"
    mod = importlib.import_module("static_ovmap.runtime_parity.views")
    request = {"scene":"room0","raw_owner":9,"final_geometry_digest":"geometry",
        "frame_id":7,"intrinsics_identity":"K","pose_identity":"pose","image_sha256":"rgb",
        "depth_sha256":"depth","target_mask_sha256":"mask","canonical_bbox":[1,2,12,14],
        "legacy_native_bbox":[1,2,11,13],"visible_pixels":110,"selection_rank":0,
        "projector_identity":"reference","request_id":"old"}
    first = mod.scientific_key(request,[1,2],"model",[20,30])
    assert first == mod.scientific_key({**request,"projector_identity":"optimized","request_id":"new"},[1,2],"model",[20,30])
    assert first != mod.scientific_key({**request,"frame_id":8},[1,2],"model",[20,30])
    assert first != mod.scientific_key(request,[2,1],"model",[20,30])


def test_cold_context_rejects_parent_recovery_handles(tmp_path):
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/runner.py"
    assert path.is_file(), "isolated cold adapter is not implemented"
    mod = importlib.import_module("static_ovmap.runtime_parity.runner")
    with pytest.raises(TypeError):
        mod.ColdContext(output_root=tmp_path,implementation="R0_REFERENCE",parent_vectors={"old":"cached"})
    context = mod.ColdContext(output_root=tmp_path,implementation="R0_REFERENCE")
    assert not any(name in vars(context) for name in ("parent_vectors","reference_outputs","selected_views","BVH"))


def test_source_parity_rejects_argmax_flip_even_with_tolerable_scores():
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/parity.py"
    assert path.is_file(), "source parity is not implemented"
    mod = importlib.import_module("static_ovmap.runtime_parity.parity")
    row = {"available":True,"label":1,"scores":[.500001,.5],"reason":"ORIGINAL_FC_STATIC_AREA",
        "attempted_request_ids":["old"],"used_request_ids":["old"],"failed_request_ids":[]}
    reference = {"valid_ids":[1,2],"objects":{"9":row}}
    selected = {"valid_ids":[1,2],"objects":{"9":{**row,"label":2,"scores":[.5,.500001],
        "attempted_request_ids":["new"],"used_request_ids":["new"]}}}
    with pytest.raises(ValueError,match="label"):
        mod.compare_source_objects(reference,selected,{"old":"new"})


def test_pilot_selection_uses_both_repeats_equal_scenes_and_simplicity_band():
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/workflow.py"
    assert path.is_file(), "bounded pilot selection is not implemented"
    mod = importlib.import_module("static_ovmap.runtime_parity.workflow")
    rows = []
    for variant,times in {"R0_REFERENCE":[10.,10.],"R1_IO_EXACT":[9.7,9.9],"R2_ROI_EXACT":[1.,18.4]}.items():
        for scene in ("room0","office1"):
            for repeat,seconds in enumerate(times,1):
                rows.append({"phase":"pilot","implementation":variant,"scene":scene,"round":repeat,
                    "status":"COMPLETE","seconds":seconds,"parity":{"status":"PASS"}})
    result = mod.select_pilot(rows,["room0","office1"],list(dict.fromkeys(row["implementation"] for row in rows)))
    assert result["selected"] == "R1_IO_EXACT"
    assert result["means"]["R2_ROI_EXACT"] == pytest.approx(9.7)
    with pytest.raises(ValueError,match="two"):
        mod.select_pilot(rows[:-1],["room0","office1"],["R2_ROI_EXACT"])


def test_final_order_reverses_scenes_and_arms_without_reusing_pilot_ids():
    path = Path(__file__).parents[2] / "src/static_ovmap/runtime_parity/workflow.py"
    assert path.is_file(), "bounded final ordering is not implemented"
    mod = importlib.import_module("static_ovmap.runtime_parity.workflow")
    spec = {"cohorts":{"replica8":["office0","room0"]},"pilot_scenes":["room0","office1"]}
    plan = mod.call_plan(spec,"final","R1_IO_EXACT")
    assert len(plan) == 16
    assert (plan[0]["scene"],plan[0]["label"]) == ("office0","U2_CONTROL")
    assert (plan[8]["scene"],plan[8]["label"]) == ("room0","G3_SELECTED")
    assert len({(row["scene"],row["label"],row["round"]) for row in plan}) == 16


def test_cached_export_verification_requires_separate_context():
    mod = importlib.import_module("static_ovmap.runtime_parity.verification")
    with pytest.raises(TypeError,match="VerificationContext"):
        mod.cached_exports(SimpleNamespace(reference={}),None,None,None,None)


def test_complete_timing_aggregate_uses_all_repeats_and_rejects_missing_cells():
    mod = importlib.import_module("static_ovmap.runtime_parity.reporting")
    scenes = ["office"+str(i) for i in range(8)]
    rows = [{"scene":scene,"round":repeat,"label":"G1_REFERENCE","status":"COMPLETE",
        "seconds":float(repeat+1),"identity":scene+str(repeat),"parity":{"status":"PASS"},
        "stage_seconds":{"recognition":float(repeat+1)},"CUDA_service_seconds_non_additive":{},
        "work_counters":{},"peak_cuda_allocated_bytes":1,"peak_cuda_reserved_bytes":2}
        for scene in scenes for repeat in (1,2)]
    result = mod.aggregate_times(rows,scenes,"G1_REFERENCE")
    assert result["mean_seconds"] == 2.5
    assert len(result["measurement_ids"]) == 16
    with pytest.raises(ValueError,match="two"):
        mod.aggregate_times(rows[:-1],scenes,"G1_REFERENCE")


def test_repeated_rgb_reuse_keeps_original_loader_bbox_and_mask(tmp_path):
    import cv2
    from static_ovmap.cvpr_compact.area_fallback_experiment import seal
    from static_ovmap.cvpr_compact.projected_views import ProjectedRegionRequest,load_projected_request
    from static_ovmap.module_validation.contracts import atomic_write_json
    from static_ovmap.module_validation.native_capture import _array_digest,_write_npz
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex
    from static_ovmap.runtime_parity.binding import file_identity
    from static_ovmap.runtime_parity.instrumentation import Stages
    from static_ovmap.runtime_parity.views import CallLoader
    image = np.arange(20*20*3,dtype=np.uint8).reshape(20,20,3)
    assert cv2.imwrite(str(tmp_path/"rgb.png"),image)
    _write_npz(tmp_path/"depth.npz",{"depth_m":np.full((20,20),2.,np.float32)})
    frame = {"frame_id":3,"image_size_hw":[20,20],"intrinsics":np.eye(3).tolist(),"pose_c2w":np.eye(4).tolist(),
        "rgb_path":"rgb.png","depth_path":"depth.npz"}
    capture_path = tmp_path/"capture.json"
    atomic_write_json(capture_path,seal({"scene_id":"fixture","frames":[frame]}))
    requests = {}
    for owner,start in ((9,2),(10,5)):
        mask = np.zeros((20,20),bool);mask[start:start+10,3:13] = True
        request = ProjectedRegionRequest("fixture","geometry",owner,3,_array_digest(np.eye(3)),_array_digest(np.eye(4)),
            file_identity(tmp_path/"rgb.png")["sha256"],file_identity(tmp_path/"depth.npz")["sha256"],_array_digest(mask),
            (3,start,13,start+10),(3,start,12,start+9),100,0,"producer")
        mask_path = tmp_path/(request.request_id+".npz")
        _write_npz(mask_path,{"packed":np.packbits(mask.ravel(),bitorder="little"),"shape":np.array(mask.shape)})
        requests[request.request_id] = {**request.to_dict(),"mask":file_identity(mask_path),"capture_manifest":str(capture_path)}
    path = tmp_path/"manifest.json"
    atomic_write_json(path,seal({"scene_id":"fixture","capture_manifest":str(capture_path),"requests":requests}))
    stages = Stages(); loader = CallLoader(path,ConsumptionIndex(),stages,reuse=True)
    for rid in requests:
        new,old = loader(rid),load_projected_request(path,rid)
        assert list(new["bbox"])==list(old["bbox"])
        for field in ("image","target","union"):
            np.testing.assert_array_equal(new[field],old[field])
    assert stages.counters["RGB_decodes"]==1
