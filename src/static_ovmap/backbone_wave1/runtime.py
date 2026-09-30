"""Native build and bounded resource execution for the isolated study."""

import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from static_ovmap.module_validation.boundary_jobs import file_identity
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest


NATIVE_V10_RECEIPT = Path("/mnt/shared/ww/ovimap-module-validation-v1/tooling/repairs-20260922/native-query-v10/receipt.json")
OLD_UPSTREAM = "/home/ww/crove/ovimap-module-validation-upstream"
OLD_BUILD_ROOT = str(NATIVE_V10_RECEIPT.parent)


@contextlib.contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"active study resource lock: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute(argv, cwd, log, *, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        number = 1
        while log.with_name(f"{log.stem}.attempt_{number:03d}{log.suffix}").exists():
            number += 1
        log.rename(log.with_name(f"{log.stem}.attempt_{number:03d}{log.suffix}"))
    start = time.monotonic()
    with log.open("wb") as stream:
        result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    row = {"argv": [str(x) for x in argv], "cwd": str(cwd), "log": str(log),
           "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - start}
    command_receipt = log.with_suffix(".commands.json")
    previous = json.loads(command_receipt.read_text()) if command_receipt.is_file() else []
    atomic_write_json(command_receipt, previous + [row])
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; inspect {log}")
    return row


def build_native(spec, repo_root, *, resume=False):
    root, upstream = Path(spec["native_build_root"]), Path(spec["upstream_worktree"])
    base = upstream / "mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm"
    wrapper = base / "src/global_segment_map_py.cpp"
    integrator = base / "src/label_tsdf_confidence_integrator.cpp"
    sources = [wrapper, integrator, base / "include/consistent_mapping/global_segment_map_py.h",
               base / "include/consistent_mapping/label_tsdf_confidence_integrator.h"]
    inputs = [file_identity(p) for p in sources] + [file_identity(NATIVE_V10_RECEIPT),
              file_identity(Path(repo_root) / spec["base_native_patch"]), file_identity(Path(__file__)),
              file_identity(Path(repo_root) / "third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch")]
    identity = canonical_digest({"inputs": inputs, "upstream_commit": spec["upstream_commit"]})
    receipt = root / "native_build_receipt.json"
    if resume and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get("identity") == identity and saved.get("status") == "COMPLETE":
            if file_identity(saved["extension"]["path"]) == saved["extension"]:
                return saved
    build = root / "build"
    build.mkdir(parents=True, exist_ok=True)
    (root / "lib").mkdir(exist_ok=True)
    old = json.loads(NATIVE_V10_RECEIPT.read_text())
    translate = lambda arg: arg.replace(OLD_UPSTREAM, str(upstream)).replace(OLD_BUILD_ROOT, str(root))
    wrapper_cmd = [translate(x) for x in old["commands"][0]["argv"]]
    wrapper_cmd[wrapper_cmd.index("-c") + 1] = str(wrapper)
    integrator_cmd = list(wrapper_cmd)
    integrator_cmd[integrator_cmd.index("-c") + 1] = str(integrator)
    integrator_cmd[integrator_cmd.index("-o") + 1] = str(build / "label_tsdf_confidence_integrator.cpp.o")
    integrator_cmd[integrator_cmd.index("-MF") + 1] = str(build / "label_tsdf_confidence_integrator.cpp.o.d")
    visualizer_cmd = [translate(x) for x in old["commands"][1]["argv"]]
    link_cmd = [translate(x) for x in old["commands"][2]["argv"]]
    link_cmd.insert(1, str(build / "label_tsdf_confidence_integrator.cpp.o"))
    commands = []
    start = time.monotonic()
    with exclusive_lock(root / ".build.lock"):
        try:
            for name, argv in [("wrapper", wrapper_cmd), ("integrator", integrator_cmd),
                               ("visualizer", visualizer_cmd), ("link", link_cmd)]:
                commands.append(execute(argv, build, root / "logs" / f"{name}.log"))
            extensions = list((root / "lib").glob("consistent_gsm*.so"))
            if len(extensions) != 1:
                raise RuntimeError("build must produce exactly one isolated extension")
            extension = extensions[0]
            env = dict(os.environ, PYTHONPATH=str(extension.parent), CUDA_VISIBLE_DEVICES="")
            script = ("import consistent_gsm as m; print(m.__file__); "
                      "assert all(hasattr(m.GlobalSegmentMap_py,k) for k in "
                      "['exportAssociationProbe','beginBackboneAssociation','configureBackboneAssociation'])")
            commands.append(execute([spec["runtime_default"], "-c", script], root,
                                    root / "logs/import.log", env=env))
            row = {"status": "COMPLETE", "identity": identity, "inputs": inputs,
                   "commands": commands, "extension": file_identity(extension),
                   "elapsed_seconds": time.monotonic() - start,
                   "upstream_commit": spec["upstream_commit"], "inherited_receipt": str(NATIVE_V10_RECEIPT),
                   "old_binary_modified": False}
            atomic_write_json(receipt, row)
            return row
        except Exception as exc:
            atomic_write_json(root / "build_failure.json", {"status": "FAILED", "error": str(exc),
                              "identity": identity, "commands": commands, "inputs": inputs})
            raise


def check_gpu_once(gpu, receipt):
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    devices = {row.split(",")[0].strip(): row.split(",")[1].strip()
               for row in inventory.stdout.splitlines()}
    if str(gpu) not in devices:
        raise ValueError("configured GPU is absent")
    occupants = [row for row in query.stdout.splitlines() if row.split(",")[0].strip() == devices[str(gpu)]]
    row = {"gpu": str(gpu), "status": "RESOURCE_BLOCK" if occupants else "AVAILABLE",
           "occupants": occupants, "inventory": inventory.stdout.splitlines(),
           "checked_at_unix": time.time(), "policy": "ONE_BOUNDED_CHECK"}
    atomic_write_json(receipt, row)
    return row
