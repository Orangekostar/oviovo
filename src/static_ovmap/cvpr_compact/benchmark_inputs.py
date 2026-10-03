"""Only the fixed CF18 captures, using the already authorized official layout."""

from dataclasses import asdict
import fcntl
from pathlib import Path
import shutil
import subprocess
import time

from static_ovmap.module_validation.assets import native_schedule
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_download import DownloaderLayout, REQUIRED_SUFFIXES, remote_identity
from static_ovmap.module_validation.scannet_frames import export_sensor, sensor_header
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read


def fixed_download_plan(spec, layout, data_root, existing=None, *, label_map_path=None):
    captures = []
    existing = existing or {}
    for scene in spec["cohorts"]["scannet_cf18"]:
        captures.append({"scene_id": scene, "physical_family": scene.split("_")[0], "files": {
            suffix: {"url": layout.scene_url(scene, suffix), "path": existing.get(scene, {}).get(suffix,
                str(Path(data_root) / "scans" / scene / (scene + suffix)))} for suffix in REQUIRED_SUFFIXES}})
    result = {"task_id": spec["task_id"], "cohort": "scannet_cf18", "captures": captures,
        "layout": asdict(layout), "label_map": {"url": layout.label_url,
            "path": str(label_map_path or (Path(data_root) / layout.label_maps[0]))},
        "authorization": "EXISTING_USER_AUTHORIZED_OFFICIAL_DOWNLOAD_NO_NEW_TERMS_ACCEPTANCE",
        "alternative_captures_allowed": False}
    result["identity"] = canonical_digest(result)
    return result


def _transfer(item, index, writable_root):
    path, url = Path(item["path"]), item["url"]
    remote = remote_identity(url)
    if path.is_file():
        if path.stat().st_size != remote["size_bytes"]:
            raise ValueError("existing raw file differs from official remote length: " + str(path))
        receipt_path = Path(str(path) + ".download.json")
        if receipt_path.exists():
            receipt = read(receipt_path)
            if any(receipt.get(k) != remote.get(k) for k in ("url", "size_bytes", "etag", "last_modified")):
                raise ValueError("existing official raw input validators changed: " + str(path))
            return {**remote, **index.identity(path, receipt), "reused": True}
        return {**remote, **index.identity(path), "reused": True,
                "validation": "OFFICIAL_REMOTE_LENGTH_LOCAL_SHA256_FORMAT_CHECK_FOLLOWS"}
    if not path.resolve().is_relative_to(Path(writable_root).resolve()):
        raise ValueError("missing input cannot be downloaded into a read-only parent root")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial, receipt_path = Path(str(path) + ".part"), Path(str(path) + ".part.json")
    if partial.exists() and (not receipt_path.is_file() or read(receipt_path) != remote):
        raise ValueError("partial raw input is bound to different remote bytes")
    atomic_write_json(receipt_path, remote)
    remaining = remote["size_bytes"] - (partial.stat().st_size if partial.is_file() else 0)
    if remaining < 0:
        raise ValueError("partial raw input exceeds official content length")
    if shutil.disk_usage(path.parent).free < remaining + 2 * 1024**3:
        raise OSError("insufficient space for the bound raw input plus 2GiB reserve")
    command = ["curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
        "--silent", "--show-error", "--connect-timeout", "15", "--max-time", "14400",
        "--speed-limit", "1024", "--speed-time", "90", "--continue-at", "-", "--output", str(partial), url]
    started = time.monotonic()
    attempt_identity = canonical_digest({"remote": remote, "command": command,
                                        "producer": index.identity(__file__)["sha256"]})
    ledger_path = Path(str(path) + ".attempts") / (attempt_identity + ".json")
    attempts = read(ledger_path)["attempts"] if ledger_path.is_file() else []
    if len(attempts) >= 3:
        raise RuntimeError("the identical transfer leaf exhausted its two automatic retries: " + str(path))
    for attempt in range(len(attempts), 3):
        print(f"CF18 download {path.name}, attempt {attempt + 1}, remaining {remaining / 1024**2:.1f}MiB", flush=True)
        attempts.append({"attempt": attempt + 1, "status": "RUNNING",
                         "partial_bytes": partial.stat().st_size if partial.exists() else 0})
        atomic_write_json(ledger_path, {"identity": attempt_identity, "command": command, "attempts": attempts})
        result = subprocess.run(command, check=False) if remaining else None
        code = result.returncode if result else 0
        attempts[-1].update(status="COMPLETE" if code == 0 else "FAILED", returncode=code,
                            partial_bytes=partial.stat().st_size if partial.exists() else 0)
        atomic_write_json(ledger_path, {"identity": attempt_identity, "command": command, "attempts": attempts,
            "elapsed_seconds": time.monotonic() - started})
        if code == 0:
            break
        if code not in {5, 6, 7, 18, 22, 28, 35, 52, 55, 56} or attempt == 2:
            raise subprocess.CalledProcessError(code, command)
        if remote_identity(url) != remote:
            raise ValueError("official remote raw input changed during a retry")
        remaining = remote["size_bytes"] - (partial.stat().st_size if partial.exists() else 0)
        time.sleep(2)
    if not partial.exists() or partial.stat().st_size != remote["size_bytes"] or remote_identity(url) != remote:
        raise ValueError("official raw input transfer is incomplete or validators changed")
    partial.rename(path)
    receipt = {**remote, **index.identity(path), "reused": False,
        "validation": "HTTPS_REMOTE_LENGTH_VALIDATORS_LOCAL_SHA256; NO_PUBLISHED_SERVER_CHECKSUM",
        "elapsed_seconds": time.monotonic() - started, "attempts": attempts}
    atomic_write_json(Path(str(path) + ".download.json"), receipt)
    return receipt


def prepare_inputs(binding, spec, *, scenes=None):
    selected = list(spec["cohorts"]["scannet_cf18"] if scenes is None else scenes)
    if len(set(selected)) != len(selected) or not set(selected) <= set(spec["cohorts"]["scannet_cf18"]):
        raise ValueError("input preparation may only consume the fixed CF18 captures")
    root = Path(binding["output_root"])
    data_root = root / "data/scannet"
    layout = DownloaderLayout(**binding["download_layout"])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    index.identity(layout.script_path, {"sha256": layout.script_sha256})
    existing = {scene: {suffix: item["path"] for suffix, item in binding["scenes"][scene]["raw_files"].items()}
                for scene in spec["cohorts"]["scannet_cf18"]}
    parent_label = Path(binding["scenes"][spec["smoke_scene"]]["runtime"]["data_root"]) / layout.label_maps[0]
    plan = fixed_download_plan(spec, layout, data_root, existing,
                               label_map_path=parent_label if parent_label.is_file() else None)
    plan_path = root / "prepare/fixed_download_plan.json"
    if plan_path.exists() and read(plan_path)["identity"] != plan["identity"]:
        raise ValueError("fixed CF18 acquisition plan changed")
    atomic_write_json(plan_path, plan)
    results = {}
    with (root / ".prepare.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        label = _transfer(plan["label_map"], index, data_root)
        for row in plan["captures"]:
            scene = row["scene_id"]
            if scene not in selected:
                continue
            receipt_path = root / "prepare" / scene / "receipt.json"
            if receipt_path.is_file() and read(receipt_path)["status"] == "COMPLETE":
                receipt = read(receipt_path)
                if receipt["plan_identity"] != plan["identity"]:
                    raise ValueError("completed CF18 input plan changed")
                for item in receipt["raw_files"].values():
                    index.identity(item["path"], item)
                export = read(receipt["export_receipt"])
                for relative, digest in export["file_hashes"].items():
                    index.identity(Path(receipt["export_root"]) / relative, {"sha256": digest})
                results[scene] = receipt
                continue
            started = time.monotonic()
            try:
                files = {suffix: _transfer(item, index, data_root) for suffix, item in row["files"].items()}
                sensor_path = Path(files[".sens"]["path"])
                with sensor_path.open("rb") as stream:
                    header = sensor_header(stream)
                schedule = native_schedule(header["source_frame_count"])
                output = data_root / "exported" / scene
                export = export_sensor(sensor_path, output, schedule)
                receipt = {"status": "COMPLETE", "scene": scene, "cohort": "scannet_cf18",
                    "physical_family": row["physical_family"], "plan_identity": plan["identity"],
                    "raw_files": files, "label_map": label, "export_root": str(output),
                    "export_receipt": str(output / "export_receipt.json"), "schedule": schedule,
                    "invalid_pose_frame_ids": export["invalid_pose_frame_ids"],
                    "sensor_source_frame_count": header["source_frame_count"], "no_invalid_slot_refill": True,
                    "elapsed_seconds": time.monotonic() - started}
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                receipt = {"status": "BLOCKED_INPUT_LEAF", "scene": scene, "plan_identity": plan["identity"],
                    "error_type": type(error).__name__, "error": str(error), "capture_not_dropped": True,
                    "elapsed_seconds": time.monotonic() - started}
            receipt["identity"] = canonical_digest(receipt)
            atomic_write_json(receipt_path, receipt)
            results[scene] = receipt
            index.write_memo(root / "validation/input_verifications.json")
            print(f"CF18 prepare {scene}: {receipt['status']}", flush=True)
    index.write_memo(root / "validation/input_verifications.json")
    coverage = {"required": len(spec["cohorts"]["scannet_cf18"]), "complete": sum(
        (root / "prepare" / scene / "receipt.json").exists() and read(root / "prepare" / scene / "receipt.json")["status"] == "COMPLETE"
        for scene in spec["cohorts"]["scannet_cf18"]), "processed_captures": selected, "results": results}
    atomic_write_json(root / "prepare/coverage.json", coverage)
    return coverage
