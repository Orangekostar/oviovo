"""Bounded real-native validation before any full recovery reconstruction."""

import argparse
import os
from pathlib import Path

from static_ovmap.backbone_wave1.runtime import execute
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import ConsumptionIndex, read


def _physical_state(state):
    keys = ("local_index", "input_instance_label", "semantic_label", "registered_label", "point_count")
    return {"segments": [{key: row[key] for key in keys} for row in state["segments"]],
            "aliases": state["aliases"], "label_instances": state["label_instances"],
            "integrated_frame_count": state["integrated_frame_count"]}


def validate(binding, build):
    repo = Path(binding["repository_root"])
    root = Path(binding["output_root"]) / "validation/native"
    upstream = Path(build["upstream_worktree"])
    scene = binding["datasets"]["development"][0]
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    results = {}
    for mode in ("off-parent", "off", "ALL_NATIVE", "A1", "A2", "A3", "follower"):
        extension = binding["scenes"][scene]["native_extension"]["path"] if mode == "off-parent" else build["extension"]["path"]
        output = root / mode
        path = output / "receipt.json"
        if path.is_file():
            receipt = read(path)
            if receipt["status"] != "COMPLETE" or receipt["native_extension"]["sha256"] != index.identity(extension)["sha256"]:
                raise ValueError("completed native validation belongs to another extension")
            for row in receipt["inputs"]:
                index.identity(row["path"], row)
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            env["PYTHONPATH"] = ":".join([str(Path(extension).parent), str(upstream / "scripts"),
                                           str(repo / "src"), str(repo)])
            env["LD_LIBRARY_PATH"] = ":".join([str(Path(extension).parent),
                str(Path(binding["scenes"][scene]["inherited"]["runtime"]["baseline_build"]) / "mapping_ros_ws/devel/lib"),
                "/home/ww/miniconda3/envs/ovimap-map/lib"])
            execute([read(binding["spec"])["default_python"], "-m", "static_ovmap.recovery_wave2.native_trace",
                "--binding", Path(binding["output_root"]) / "resolved_inputs.json", "--scene", scene,
                "--mode", mode, "--extension", extension, "--output", output], repo, root / (mode + ".log"), env=env)
            receipt = read(path)
        results[mode] = receipt
    for mode in ("off", "ALL_NATIVE"):
        reference, current = results["off-parent"], results[mode]
        if reference["tsdf_array_identities"] != current["tsdf_array_identities"]:
            raise ValueError("serial native/fallback changed the actual TSDF arrays")
        if [_physical_state(row["state"]) for row in reference["frames"]] != [
                _physical_state(row["state"]) for row in current["frames"]]:
            raise ValueError("serial native/fallback changed labels, counts, owners or aliases")
        if [row["owner_projection_sha256"] for row in reference["frames"]] != [
                row["owner_projection_sha256"] for row in current["frames"]]:
            raise ValueError("serial native/fallback changed the actual owner raster")
    follower = results["follower"]["frames"][-1]
    if len(follower["plan"]["accepted_pairs"]) != 2 or follower["plan"]["duplicate_owner_followers"] != 1:
        raise ValueError("native follower fixture did not exercise compatible duplicate owners")
    accepted = sum(len(row["plan"]["accepted_pairs"]) for mode in ("A1", "A2", "A3")
                   for row in results[mode]["frames"])
    receipt = {"status": "VERIFIED", "new_extension": build["extension"],
        "native_off_vs_parent_exact": True, "all_USE_NATIVE_vs_parent_exact": True,
        "three_actual_valid_frames": [row["frame_id"] for row in results["off"]["frames"]],
        "accepted_real_frame_pairs": accepted, "actual_native_follower_verified": True,
        "native_follower_groups": follower["plan"]["accepted_pairs"],
        "native_follower_segments": follower["state"]["segments"],
        "receipts": {mode: index.identity(root / mode / "receipt.json") for mode in results},
        "new_visual_inference": 0, "GT_input": False}
    receipt["identity"] = canonical_digest(receipt)
    atomic_write_json(root / "validation_receipt.json", receipt)
    print(receipt["status"], "accepted", accepted, "follower", receipt["native_follower_groups"], flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--build", required=True)
    args = parser.parse_args()
    validate(read(args.binding), read(args.build))
