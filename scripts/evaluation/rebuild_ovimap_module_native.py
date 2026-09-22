"""Rebuild the pinned capture extension and replay the bound two historical frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.static_ovmap.module_validation.boundary_jobs import (
    capture_inputs,
    file_identity,
    verify_capture,
)
from src.static_ovmap.module_validation.contracts import atomic_write_json

PIN = "f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--baseline-build",
        type=Path,
        required=True,
        help="OVI-MAP root containing ABI-compatible mapping_ros_ws build/devel",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory; existing results are never overwritten",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frontend-cache", type=Path, required=True)
    parser.add_argument("--generated-ros-headers", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    upstream, baseline, output = (
        path.resolve()
        for path in (args.upstream, args.baseline_build, args.output_root)
    )
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = (
        ROOT
        / "third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch"
    )
    difference = subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "--binary", "--unified=50", "HEAD"]
    )
    if (
        head != PIN
        or hashlib.sha256(difference).hexdigest() != file_identity(patch)["sha256"]
    ):
        raise ValueError("upstream must match the pinned commit and exact study patch")
    output.mkdir(parents=True, exist_ok=False)
    build = output / "build"
    extension_dir = output / "lib"
    build.mkdir()
    extension_dir.mkdir()
    commands, inputs = [], [file_identity(patch)]
    environment = dict(os.environ)
    environment.update({key: "8" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM")})
    library_dirs = [
        str(extension_dir),
        str(baseline / "mapping_ros_ws/devel/lib"),
        str(Path(sys.prefix) / "lib"),
    ]
    environment["LD_LIBRARY_PATH"] = ":".join(
        library_dirs + [environment.get("LD_LIBRARY_PATH", "")]
    )
    environment["PYTHONPATH"] = ":".join(
        [
            str(extension_dir),
            str(baseline / "mapping_ros_ws/devel/lib"),
            str(upstream / "scripts"),
            str(ROOT),
        ]
    )

    def run(command, cwd, log_name):
        commands.append(
            {"argv": list(map(str, command)), "cwd": str(cwd), "log": log_name}
        )
        atomic_write_json(
            output / "commands.json",
            {
                "commands": commands,
                "environment": {
                    key: environment[key] for key in ("LD_LIBRARY_PATH", "PYTHONPATH")
                },
            },
        )
        with (output / log_name).open("w") as log:
            result = subprocess.run(
                list(map(str, command)),
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"native job failed ({result.returncode}); see {output / log_name}"
            )

    # Reuse only ABI-compatible baseline dependencies. Compile study sources and
    # their inline headers from the isolated, pinned checkout.
    objects = []
    for target, source in (
        ("consistent_gsm", "src/global_segment_map_py.cpp"),
        ("global_segment_map", "src/utils/visualizer.cc"),
    ):
        flags_path = (
            baseline
            / f"mapping_ros_ws/build/{target}/CMakeFiles/{target}.dir/flags.make"
        )
        inputs.append(file_identity(flags_path))
        flags = {}
        for line in flags_path.read_text().splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                flags[key] = shlex.split(value)
        source_path = (
            upstream
            / f"mapping_ros_ws/src/consistent_panoptic_mapping/{target}/{source}"
        )
        inputs.append(file_identity(source_path))
        obj = build / f"{Path(source).name}.o"
        tokens = flags["CXX_DEFINES"] + flags["CXX_INCLUDES"] + flags["CXX_FLAGS"]
        old = str(baseline / "mapping_ros_ws/src/consistent_panoptic_mapping")
        new = str(upstream / "mapping_ros_ws/src/consistent_panoptic_mapping")
        tokens = [
            token.replace(old, new) if "/cvnp" not in token else token
            for token in tokens
        ]
        tokens += [
            "-I"
            + str(
                upstream
                / "mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map_node/include"
            )
        ]
        tokens += [
            "-I" + str(upstream / "mapping_ros_ws/src/voxblox/voxblox_ros/include")
        ]
        tokens += [
            "-I" + str(args.generated_ros_headers.resolve()),
            "-I"
            + str(
                baseline
                / "mapping_ros_ws/src/minkindr_ros/minkindr_conversions/include"
            ),
        ]
        tokens = [
            token.replace(
                str(baseline / "mapping_ros_ws/src/voxblox"),
                str(upstream / "mapping_ros_ws/src/voxblox"),
            )
            for token in tokens
        ]
        run(
            [
                "/usr/bin/c++",
                *tokens,
                "-MMD",
                "-MF",
                str(obj) + ".d",
                "-c",
                source_path,
                "-o",
                obj,
            ],
            build,
            f"compile-{target}.log",
        )
        objects.append(obj)
    link_path = (
        baseline
        / "mapping_ros_ws/build/consistent_gsm/CMakeFiles/consistent_gsm.dir/link.txt"
    )
    inputs.append(file_identity(link_path))
    link = shlex.split(link_path.read_text())
    link = [
        f"-Wl,--dependency-file={build / 'link.d'}"
        if token.startswith("-Wl,--dependency-file=")
        else token
        for token in link
    ]
    extension = extension_dir / Path(link[link.index("-o") + 1]).name
    link[link.index("-o") + 1] = str(extension)
    for index, token in enumerate(link):
        if token.endswith("global_segment_map_py.cpp.o"):
            link[index] = str(objects[0])
        elif token.endswith(".o"):
            link[index] = str(baseline / "mapping_ros_ws/build/consistent_gsm" / token)
            inputs.append(file_identity(link[index]))
        elif (
            token != str(extension) and token.startswith("/") and token.endswith(".so")
        ):
            inputs.append(file_identity(token))
    link.append(str(objects[1]))
    visualization_library = Path(sys.prefix) / "lib/libpcl_visualization.so"
    link.append(str(visualization_library))
    inputs.append(file_identity(visualization_library))
    run(link, build, "link.log")
    run(
        [
            sys.executable,
            "-c",
            "import consistent_gsm; print(consistent_gsm.__file__); assert all(hasattr(consistent_gsm.GlobalSegmentMap_py, name) for name in ('exportStudySurfaceLabels', 'exportStudyTsdfState'))",
        ],
        output,
        "import.log",
    )
    environment["OVIMAP_MODULE_VALIDATION_CAPTURE_ROOT"] = str(output / "capture")
    environment["OVIMAP_MODULE_VALIDATION_HELPER_ROOT"] = str(ROOT)
    cache = args.frontend_cache.resolve()
    command = [
        sys.executable,
        upstream / "scripts/panoptic_mapping_.py",
        "--dataset",
        "replica",
        "--scene_num",
        "room0",
        "--result_folder",
        output / "mapper",
        "--data_folder",
        args.data_root.resolve(),
        "--start",
        "0",
        "--end",
        "20",
        "--step",
        "10",
        "--num_threads",
        "8",
        "--task",
        "Nyu40",
        "--data_association",
        "2",
        "--inst_association",
        "4",
        "--seg_graph_confidence",
        "3",
        "--use_inst_label_connect",
        "1",
        "--connection_ratio_th",
        "0.2",
        "--skip_feature_extraction",
        "--use_temp_panoptics",
        "--temp_panoptics_folder",
        cache / "sem_seg_temp/room0/cropformer",
        "--use_temp_geometrics",
        "--temp_geometrics_folder",
        cache / "geo_seg_temp/room0",
        "--intermediate_seg_folder",
        cache / "fuse_seg_temp/room0",
    ]
    run(command, upstream, "replay.log")
    manifest_path = output / "capture/room0/manifest.json"
    manifest = verify_capture(manifest_path)
    if manifest["native_extension"]["path"] != str(extension):
        raise AssertionError("replay loaded a different native extension")
    membership_counts = []
    for frame in manifest["frames"]:
        state = json.loads((manifest_path.parent / frame["native_state"]["path"]).read_text())["native_state"]
        if state.get("label_instances_scope") != "all_known_labels":
            raise AssertionError("native replay lacks full current label membership")
        labels = [int(row["segment_label"]) for row in state["label_instances"]]
        registered = {int(row["registered_label"]) for row in state["segments"]}
        if labels != sorted(set(labels)) or not registered <= set(labels):
            raise AssertionError("full native membership is unordered, duplicated, or incomplete")
        membership_counts.append({"frame_id": frame["frame_id"], "all_known_labels": len(labels),
                                  "currently_registered_labels": len(registered)})
    atomic_write_json(
        output / "receipt.json",
        {
            "artifact_type": "OVIMAP_REPRODUCIBLE_NATIVE_REPLAY",
            "status": "COMPLETE",
            "scientific_result": False,
            "elapsed_seconds": time.monotonic() - started,
            "extension": file_identity(extension),
            "capture_manifest": file_identity(manifest_path),
            "frame_ids": manifest["completed_frame_ids"],
            "current_membership": membership_counts,
            "inputs": inputs,
            "commands": commands,
            "payloads": [
                file_identity(row["path"])
                for row in capture_inputs(manifest_path, manifest)
            ],
        },
    )
    print(json.dumps({"status": "COMPLETE", "receipt": str(output / "receipt.json")}))


if __name__ == "__main__":
    main()
