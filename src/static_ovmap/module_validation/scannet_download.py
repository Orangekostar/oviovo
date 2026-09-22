"""Bounded ScanNet v2 preparation using the user's official downloader layout."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .assets import (
    _metadata_frame_count,
    _split_scene_ids,
    native_schedule,
    physical_family,
    sha256_file,
)
from .contracts import atomic_write_json, canonical_digest

SCANNET_REVISION = "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c"
PUBLIC_ROOT = f"https://raw.githubusercontent.com/ScanNet/ScanNet/{SCANNET_REVISION}/Tasks/Benchmark/"
REQUIRED_SUFFIXES = (
    ".sens", ".txt", "_vh_clean_2.ply", "_vh_clean_2.labels.ply",
    "_vh_clean_2.0.010000.segs.json", ".aggregation.json",
)


@dataclass(frozen=True)
class DownloaderLayout:
    base_url: str
    releases: tuple[str, ...]
    task_releases: tuple[str, ...]
    label_maps: tuple[str, ...]
    script_path: str
    script_sha256: str

    @classmethod
    def from_script(cls, path: Path) -> DownloaderLayout:
        """Read constants only: never execute SSL overrides, prompts or downloads."""
        names = {"BASE_URL", "RELEASES", "RELEASES_TASKS", "LABEL_MAP_FILES"}
        values = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        values[target.id] = ast.literal_eval(node.value)
        if set(values) != names:
            raise ValueError("provided downloader is missing literal release constants")
        base = values["BASE_URL"].replace("http://", "https://", 1)
        if (base != "https://kaldir.vc.cit.tum.de/scannet/"
                or values["RELEASES"] != ["v2/scans", "v1/scans"]
                or values["RELEASES_TASKS"] != ["v2/tasks", "v1/tasks"]
                or values["LABEL_MAP_FILES"] != ["scannetv2-labels.combined.tsv", "scannet-labels.combined.tsv"]):
            raise ValueError("provided downloader differs from the reviewed ScanNet v2 layout")
        return cls(base, tuple(values["RELEASES"]), tuple(values["RELEASES_TASKS"]),
                   tuple(values["LABEL_MAP_FILES"]), str(path.resolve()), sha256_file(path))

    def scene_url(self, scene_id: str, suffix: str) -> str:
        if not re.fullmatch(r"scene\d{4}_\d{2}", scene_id) or suffix not in REQUIRED_SUFFIXES:
            raise ValueError("invalid scene ID or out-of-scope ScanNet file type")
        release = self.releases[1 if suffix == ".sens" else 0]
        return f"{self.base_url}{release}/{scene_id}/{scene_id}{suffix}"

    @property
    def label_url(self) -> str:
        return f"{self.base_url}{self.task_releases[0]}/{self.label_maps[0]}"


def _ordered_families(scene_ids, exclusions):
    families = {}
    for scene in scene_ids:
        family = physical_family(scene)
        if family not in exclusions:
            families.setdefault(family, []).append(scene)
    order = sorted(families, key=lambda family: (hashlib.sha256(f"ovimap-module-v1|{family}".encode()).hexdigest(), family))
    return [{"family_id": family, "captures": sorted(families[family], key=lambda scene: (not scene.endswith("_00"), scene))}
            for family in order]


def prepare_plan(layout: DownloaderLayout, train: Path, validation: Path, exclusions) -> dict:
    train_ids, val_ids = _split_scene_ids(train), _split_scene_ids(validation)
    if {physical_family(s) for s in train_ids} & {physical_family(s) for s in val_ids}:
        raise ValueError("official split physical-family overlap")
    excluded = sorted({physical_family(s) for s in exclusions})
    ordered = {"train": _ordered_families(train_ids, excluded), "validation": _ordered_families(val_ids, excluded)}
    if len(ordered["train"]) < 12 or len(ordered["validation"]) < 2:
        raise ValueError("not enough independent physical families for the frozen 12+2 protocol")
    candidates = []
    for split, roles in (("train", ["fit"] * 8 + ["cal"] * 2 + ["select"] * 2), ("validation", ["confirm"] * 2)):
        for family, role in zip(ordered[split], roles):
            scene = family["captures"][0]
            candidates.append({"scene_id": scene, "family_id": family["family_id"], "role": role,
                               "files": {suffix: layout.scene_url(scene, suffix) for suffix in REQUIRED_SUFFIXES}})
    return {"schema_version": 1, "artifact_type": "OVIMAP_SCANNET_ACQUISITION_PLAN",
            "status": "PROVISIONAL_REQUIRES_AVAILABILITY_CHECK", "layout": asdict(layout),
            "public_source_revision": SCANNET_REVISION,
            "split_files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in (train, validation)],
            "exclusions": excluded, "ordered_families": ordered, "candidates": candidates,
            "required_file_types": list(REQUIRED_SUFFIXES),
            "terms_url": layout.base_url + "ScanNet_TOS.pdf",
            "confirmation_processing": "DEFER_UNTIL_FROZEN_SELECTION"}


def remote_identity(url: str) -> dict:
    if not url.startswith("https://"):
        raise ValueError("verified HTTPS is required")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
                if not response.url.startswith("https://"):
                    raise ValueError("HTTPS downgrade rejected")
                size = int(response.headers.get("Content-Length", "0"))
                if size <= 0:
                    raise ValueError(f"missing positive remote content length: {url}")
                return {"url": url, "size_bytes": size, "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified")}
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2)
    raise RuntimeError("remote identity unavailable")


def record_download(path: Path, remote: dict) -> dict:
    if path.stat().st_size != remote["size_bytes"]:
        raise ValueError(f"download size mismatch: {path}")
    receipt = {**remote, "path": str(path.resolve()), "sha256": sha256_file(path),
               "validation": "HTTPS_REMOTE_LENGTH_AND_LOCAL_SHA256; no published server checksum"}
    atomic_write_json(Path(str(path) + ".download.json"), receipt)
    return receipt


def verified_download(path: Path, remote: dict) -> bool:
    try:
        receipt = json.loads(Path(str(path) + ".download.json").read_text())
        return (path.stat().st_size == remote["size_bytes"]
                and all(receipt.get(k) == remote.get(k) for k in ("url", "size_bytes", "etag", "last_modified"))
                and sha256_file(path) == receipt["sha256"])
    except (OSError, KeyError, ValueError):
        return False


def transfer(url: str, path: Path, remote: dict | None = None) -> dict:
    """Resume a bound .part file; do not overwrite unrelated or corrupt finals."""
    remote = remote or remote_identity(url)
    if remote["url"] != url:
        raise ValueError("remote identity URL mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            if verified_download(path, remote):
                return json.loads(Path(str(path) + ".download.json").read_text())
            if not Path(str(path) + ".download.json").exists() and path.stat().st_size == remote["size_bytes"]:
                # Existing local data may be reused; format checks occur before export.
                return record_download(path, remote)
            raise ValueError(f"existing file failed verification; preserved for inspection: {path}")
        partial = Path(str(path) + ".part")
        binding = Path(str(path) + ".part.json")
        if partial.exists():
            if not binding.is_file() or json.loads(binding.read_text()) != remote:
                raise ValueError(f"partial download identity changed: {partial}")
            if partial.stat().st_size > remote["size_bytes"]:
                raise ValueError(f"partial file exceeds remote size: {partial}")
        atomic_write_json(binding, remote)
        remaining = remote["size_bytes"] - (partial.stat().st_size if partial.exists() else 0)
        if shutil.disk_usage(path.parent).free < remaining + 2 * 1024**3:
            raise OSError("insufficient free space for this download plus 2 GiB reserve")
        if remaining:
            print(f"Downloading {path.name}: {remaining / 1024**2:.1f} MiB remaining", flush=True)
            subprocess.run([
                "curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
                "--silent", "--show-error", "--connect-timeout", "15", "--max-time", "14400",
                "--speed-limit", "1024", "--speed-time", "90", "--retry", "3", "--retry-delay", "2",
                "--retry-all-errors", "--continue-at", "-", "--output", str(partial), url,
            ], check=True)
        if partial.stat().st_size != remote["size_bytes"]:
            raise ValueError(f"incomplete transfer: {partial}")
        # Check validators again before publishing a resumed stream.
        if remote_identity(url) != remote:
            raise ValueError(f"remote content changed during transfer: {url}")
        partial.rename(path)
        return record_download(path, remote)


def fetch_public_splits(root: Path) -> tuple[Path, Path]:
    paths = tuple(root / name for name in ("scannetv2_train.txt", "scannetv2_val.txt"))
    for path in paths:
        transfer(PUBLIC_ROOT + path.name, path)
    return paths


def download_selected(plan: dict, root: Path, *, authorized: bool) -> dict:
    if not authorized:
        raise PermissionError("explicit ScanNet download authorization and prior terms agreement required")
    layout = DownloaderLayout(**plan["layout"])
    if sha256_file(layout.script_path) != layout.script_sha256:
        raise ValueError("supplied downloader changed after planning")
    for source in plan["split_files"]:
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("official split changed after planning")
    expected = prepare_plan(DownloaderLayout.from_script(Path(layout.script_path)),
                            *(Path(row["path"]) for row in plan["split_files"]), plan["exclusions"])
    if canonical_digest(plan) != canonical_digest(expected):
        raise ValueError("acquisition plan differs from the frozen ordering or file scope")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".acquisition.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _download_locked(plan, layout, root)


def _download_locked(plan: dict, layout: DownloaderLayout, root: Path) -> dict:
    lock_path = root / "acquisition_lock.json"
    if lock_path.exists():
        locked = json.loads(lock_path.read_text())
        if locked["plan_digest"] != canonical_digest(plan):
            raise ValueError("acquisition lock belongs to a different input plan")
    else:
        selected, rejected = [], []
        for split, roles in (("train", ["fit"] * 8 + ["cal"] * 2 + ["select"] * 2), ("validation", ["confirm"] * 2)):
            accepted = 0
            for family in plan["ordered_families"][split]:
                for scene in family["captures"]:
                    try:
                        files = {suffix: remote_identity(layout.scene_url(scene, suffix)) for suffix in REQUIRED_SUFFIXES}
                    except urllib.error.HTTPError as error:
                        if error.code != 404:
                            raise
                        rejected.append({"scene_id": scene, "reason": "MISSING_REMOTE_FILE", "url": error.url})
                        continue
                    metadata = root / "scans" / scene / f"{scene}.txt"
                    transfer(files[".txt"]["url"], metadata, files[".txt"])
                    count = _metadata_frame_count(metadata)
                    if count is None:
                        raise ValueError(f"metadata has no native frame count: {metadata}")
                    if count < 200:
                        rejected.append({"scene_id": scene, "reason": "FEWER_THAN_200_NATIVE_SLOTS"})
                        continue
                    selected.append({"scene_id": scene, "family_id": family["family_id"],
                                     "role": roles[accepted], "source_split": split,
                                     "schedule": native_schedule(count), "files": files})
                    accepted += 1
                    break
                if accepted == len(roles):
                    break
            if accepted != len(roles):
                raise ValueError(f"insufficient available {split} scenes; frozen split cannot be reduced")
        locked = {"schema_version": 1, "status": "AVAILABILITY_LOCKED_NOT_YET_DOWNLOADED",
                  "plan_digest": canonical_digest(plan), "selected": selected, "rejected": rejected,
                  "total_remote_bytes": sum(f["size_bytes"] for r in selected for f in r["files"].values()),
                  "authorization": "User explicitly confirmed prior ScanNet authorization and terms agreement"}
        atomic_write_json(lock_path, locked)
    remaining = sum(
        remote["size_bytes"] - min(remote["size_bytes"],
            (root / "scans" / row["scene_id"] / f"{row['scene_id']}{suffix}").stat().st_size
            if (root / "scans" / row["scene_id"] / f"{row['scene_id']}{suffix}").is_file() else 0)
        for row in locked["selected"] for suffix, remote in row["files"].items()
    )
    if shutil.disk_usage(root).free < remaining + 2 * 1024**3:
        raise OSError("insufficient free space for the locked scene set plus 2 GiB reserve")
    for row in locked["selected"]:
        for suffix, remote in row["files"].items():
            if remote_identity(remote["url"]) != remote:
                raise ValueError("remote identity changed after availability lock")
            transfer(remote["url"], root / "scans" / row["scene_id"] / f"{row['scene_id']}{suffix}", remote)
    transfer(layout.label_url, root / layout.label_maps[0])
    receipt = {"schema_version": 1, "status": "RAW_DOWNLOADS_COMPLETE", "lock_sha256": sha256_file(lock_path),
               "scene_count": len(locked["selected"]), "total_remote_bytes": locked["total_remote_bytes"],
               "confirmation_processing": "NOT_STARTED"}
    atomic_write_json(root / "download_receipt.json", receipt)
    return receipt
