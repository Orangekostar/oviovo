"""Isolated recovery build using the measured original native compilation recipe."""

import argparse
import os
from pathlib import Path

from static_ovmap.backbone_wave1.runtime import build_native as inherited_build, execute
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import ConsumptionIndex, read
from .native_patch import export_patch


def build_native(binding, upstream, build_root, *, resume=False):
    repo = Path(binding["repository_root"])
    spec = read(binding["spec"])
    patch = export_patch(repo, upstream, build_root)
    arguments = {"upstream_worktree": str(upstream), "native_build_root": str(build_root),
        "upstream_commit": spec["upstream"]["commit"],
        "base_native_patch": "third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch",
        "runtime_default": spec["default_python"]}
    result = inherited_build(arguments, repo, resume=resume)
    extension = result["extension"]
    script = ("import consistent_gsm as m; assert all(hasattr(m.GlobalSegmentMap_py,k) for k in "
              "['configureRecoveryAssociation','beginRecoveryAssociation','exportAssociationProbe']); "
              "print(m.__file__)")
    command = execute([spec["default_python"], "-c", script], build_root,
        Path(build_root) / "logs/recovery_import.log", env=dict(os.environ,
            PYTHONPATH=str(Path(extension["path"]).parent), CUDA_VISIBLE_DEVICES=""))
    index = ConsumptionIndex()
    layers = [index.identity(repo / "third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch"),
              index.identity(repo / "third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch"),
              index.identity(patch)]
    receipt = {**result, "recovery_action_import": command, "patch_stack": layers,
               "recovery_runtime": index.identity(__file__), "parent_binary_modified": False,
               "new_extension": extension, "upstream_worktree": str(upstream)}
    receipt["recovery_identity"] = canonical_digest({"build_identity": result["identity"], "patches": layers,
                                                    "runtime": receipt["recovery_runtime"]})
    atomic_write_json(Path(build_root) / "recovery_native_build_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = build_native(read(args.binding), args.upstream, args.build_root, resume=args.resume)
    print(result["status"], result["extension"], flush=True)
