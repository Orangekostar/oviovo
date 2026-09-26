"""New-only compact evidence, ordinary task-branch push, external SHA receipt."""

import gzip
import subprocess
from pathlib import Path

from src.static_ovmap.module_validation.contracts import atomic_write_json
from src.static_ovmap.module_validation.scannet_study import load_prediction

from .io import ROOT, SourceIndex, read_json, write_once
from .object_evidence import owner_labels


def export_artifacts(config):
    root = Path(config["attempt_root"])
    report = read_json(root / "report_receipt.json")
    tables = read_json(report["tables"])
    destination = ROOT / "artifacts/static_ovmap/complementary_composition_v1"
    destination.mkdir(parents=True, exist_ok=True)
    index = SourceIndex()
    compact = []

    def copy_new(path, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = Path(path).read_bytes()
        compressed = len(content) > 256 * 1024 or target.suffix == ".gz"
        if compressed and target.suffix != ".gz":
            target = target.with_name(target.name + ".gz")
        if target.is_file():
            previous = (
                gzip.decompress(target.read_bytes())
                if compressed
                else target.read_bytes()
            )
            if previous != content:
                raise ValueError(f"immutable compact artifact changed: {target}")
        else:
            target.write_bytes(
                gzip.compress(content, mtime=0) if compressed else content
            )
        compact.append(
            {"source": index.identity(path), "artifact": index.identity(target)}
        )

    for name in (
        "binding.json",
        "source_manifest.json",
        "resolved_config.json",
        "selection.json",
    ):
        if (root / name).is_file():
            copy_new(root / name, Path("binding") / name)
    for folder in ("calibration", "rows", "sources", "query", "compositions"):
        for path in sorted((root / folder).rglob("*.json")):
            if path.name == "resource_status.json" or "prediction" in path.parts:
                continue
            copy_new(path, path.relative_to(root))
    # Confirmation traces use their own immutable sub-attempt roots.
    for path in sorted((root / "confirmation").glob("*/resolved_config.json")):
        scene_root = path.parent
        for name in ("resolved_config.json", "source_manifest.json"):
            copy_new(scene_root / name, (scene_root / name).relative_to(root))
        for folder in ("query", "sources", "compositions"):
            for item in sorted((scene_root / folder).rglob("*.json")):
                if (
                    item.name != "resource_status.json"
                    and "prediction" not in item.parts
                ):
                    copy_new(item, item.relative_to(root))
    for name in ("access.json", "receipt.json", "exposure_block.json"):
        if (root / "confirmation" / name).is_file():
            copy_new(root / "confirmation" / name, Path("confirmation") / name)
    for path in sorted((root / "technical").rglob("*.json")):
        copy_new(path, path.relative_to(root))
    # Report generations are content-addressed; a partial publication is retained.
    copy_new(
        Path(report["tables"]),
        Path("reports") / Path(report["tables"]).parent.name / "tables.json",
    )
    for row in tables["rows"]:
        payload = load_prediction(row["prediction_manifest"])
        local = config
        if row["role"] == "confirmation":
            local = read_json(
                root / "confirmation" / row["scene_id"] / "resolved_config.json"
            )
        compact_labels = {
            "scene_id": row["scene_id"],
            "method_id": payload.method_id,
            "branch": payload.branch,
            "labels": owner_labels(payload),
            "logical_cost": dict(payload.logical_cost),
            "metadata": dict(payload.metadata),
            "prediction_key": payload.prediction_key,
            "record_key": payload.record_key,
            "native_manifest": index.identity(
                local["scenes"][row["scene_id"]]["native_prediction"]
            ),
        }
        label_path = (
            destination
            / "labels"
            / row["role"]
            / row["scene_id"]
            / (row["method_id"] + ".json")
        )
        write_once(label_path, compact_labels)
        for key in ("trace_path", "matches_path"):
            trace = Path(row[key])
            target = destination / "released" / row["evaluation_identity"] / trace.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                if target.read_bytes() != trace.read_bytes():
                    raise ValueError("released trace compact copy changed")
            else:
                target.write_bytes(
                    trace.read_bytes()
                )  # Already compressed once by the evaluator.
    # Only new output trees are inventoried, plus the explicitly bound historical inputs.
    large = [
        index.identity(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix
        in {".npz", ".npy", ".ply", ".pkl", ".png", ".jpg", ".sens", ".log"}
    ]
    atomic_write_json(
        destination / "external_artifacts.json",
        {
            "new_outputs": large,
            "historical_sources": read_json(root / "source_manifest.json"),
            "native_v10_commit": config["reviewed_commit"],
            "no_previous_release_tree_copied": True,
        },
    )
    atomic_write_json(
        destination / "compact_manifest.json",
        {
            "status": report["status"],
            "artifacts": compact,
            "tables": str(Path(report["tables"]).parent.name),
        },
    )
    commands = (
        "# Reconstruct or resume\n\nDense predictions reuse their exact immutable N0 geometry and ranks.\n\n```bash\n"
        f"{config['runtime']['native_perception_python']} -m src.static_ovmap.composition_study.reconstruction --labels artifacts/static_ovmap/complementary_composition_v1/labels/compose_cal/scene0056_00/N0.json --output /mnt/shared/ww/ovimap-complementary-composition-v1/reconstruction/scene0056_00/N0\n"
        f"{config['runtime']['native_perception_python']} scripts/evaluation/run_ovimap_composition_study.py --resolved-config {root / 'resolved_config.json'} --phase all\n```\n\n"
        "Use each labels/<role>/<scene>/<method>.json for the corresponding exact prediction; hashes and record keys are checked. External inputs are listed in external_artifacts.json. No data/model download is part of reconstruction.\n"
    )
    (destination / "RECONSTRUCTION.md").write_text(commands)
    return destination


def publish(config):
    root = Path(config["attempt_root"])
    report = read_json(root / "report_receipt.json")
    destination = export_artifacts(config)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != config["task_branch"]:
        raise ValueError("publication must use the new task branch")
    allowed = [
        "src/static_ovmap/composition_study",
        "tests/composition_study",
        "scripts/evaluation/run_ovimap_composition_study.py",
        "configs/evaluation/ovimap_composition_study.json",
        str(destination.relative_to(ROOT)),
        "docs/paper/static_ovmap/COMPOSITION_RESULTS.md",
        "docs/paper/static_ovmap/COMPOSITION_HANDOFF.md",
        "docs/paper/static_ovmap/complementary_composition_v1",
        "docs/superpowers/plans/2026-09-26-complementary-composition.md",
    ]
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
    ).splitlines()
    if any(
        not any(path == prefix or path.startswith(prefix + "/") for prefix in allowed)
        for path in staged
    ):
        raise ValueError("unrelated staged work must not enter composition publication")
    subprocess.run(["git", "add", "--", *allowed], cwd=ROOT, check=True)
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
    ).returncode:
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "Publish complementary composition evidence (" + report["status"] + ")",
            ],
            cwd=ROOT,
            check=True,
        )
    local = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    try:
        subprocess.run(
            ["git", "push", "-u", "origin", "HEAD:refs/heads/" + branch],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        remote_line = subprocess.check_output(
            ["git", "ls-remote", "--heads", "origin", "refs/heads/" + branch],
            cwd=ROOT,
            text=True,
        ).strip()
        remote = remote_line.split()[0] if remote_line else None
        if remote != local:
            raise RuntimeError("remote branch SHA differs after ordinary push")
        result = {
            "status": "PUSH_VERIFIED",
            "release_commit_B": local,
            "remote_sha": remote,
            "branch": branch,
            "experiment_status": report["status"],
            "report_receipt": str(root / "report_receipt.json"),
        }
    except (subprocess.CalledProcessError, RuntimeError) as error:
        result = {
            "status": "PUSH_FAILED",
            "local_sha": local,
            "branch": branch,
            "error": str(error),
            "experiment_status": report["status"],
        }
    atomic_write_json(root / "publication_receipt.json", result)
    return result
