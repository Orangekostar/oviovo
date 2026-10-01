"""Archive byte-exact consumed Python producers, including the pre-commit wrapper."""

import gzip
import hashlib
from pathlib import Path
import subprocess

from static_ovmap.module_validation.contracts import atomic_write_json

from .binding import ConsumptionIndex


EARLY_FC_SHA = "7723aef12433579efe4bdc30a8d3b1337bf1ec7301ad1947c3d9aa493080495f"


def _early_fc(repo):
    value = subprocess.check_output(["git", "show", "2d8bd2ec:src/static_ovmap/recovery_wave2/recovery_fc_worker.py"], cwd=repo).decode()
    start = value.index('                            foreign =')
    end = value.index('                            load_begin =', start)
    value = value[:start] + ('                            if resource["status"] != "AVAILABLE":\n'
        '                                raise RuntimeError("RESOURCE_BLOCK: recovery FC GPU is occupied")\n') + value[end:]
    value = value.replace("import os\n", "").replace(
        '                    index.identity(capture_path.parent / frame[name],\n'
        '                                   {"sha256": frame[name.replace("_path", "_sha256")]})',
        '                    index.identity(capture_path.parent / frame[name])')
    data = value.encode()
    if hashlib.sha256(data).hexdigest() != EARLY_FC_SHA:
        raise ValueError("historical wrapper restoration failed its original consumed-byte checksum")
    return data


def archive_producers(binding, rows):
    repo = Path(binding["repository_root"]).resolve()
    destination = repo / "artifacts/static_ovmap/recovery_wave2_v1/validation/producer_sources"
    index, results, history, seen = ConsumptionIndex(), [], {}, set()
    for row in rows:
        path = Path(row["path"]).resolve()
        if path.suffix != ".py" or not path.is_relative_to(repo):
            continue
        key = (str(path), row["sha256"])
        if key in seen:
            continue
        seen.add(key)
        relative, data, commit = str(path.relative_to(repo)), None, None
        if row["sha256"] == EARLY_FC_SHA:
            data = _early_fc(repo)
            origin = "EXACT_PRECOMMIT_SOURCE_RESTORED_BY_REVERSING_RECORDED_RESOURCE_AND_HASH_GUARD_PATCH"
        else:
            current = path.read_bytes()
            if hashlib.sha256(current).hexdigest() == row["sha256"]:
                data, origin = current, "CURRENT_EXACT_CONSUMED_BYTES"
            else:
                if relative not in history:
                    history[relative] = subprocess.check_output(["git", "log", "--format=%H", "--", relative], cwd=repo, text=True).splitlines()
                for candidate in history[relative]:
                    blob = subprocess.check_output(["git", "show", candidate + ":" + relative], cwd=repo)
                    if hashlib.sha256(blob).hexdigest() == row["sha256"]:
                        data, commit, origin = blob, candidate, "EXACT_COMMITTED_GIT_BLOB"
                        break
        if data is None:
            raise ValueError("consumed Python producer cannot be restored byte-exactly: " + relative + " " + row["sha256"])
        if len(data) != row.get("bytes", len(data)):
            raise ValueError("restored producer length differs from the consumed receipt")
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / (row["sha256"] + ".py.gz")
        with archive.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            zipped.write(data)
        results.append({"original_path": row["path"], "relative_repository_path": relative,
            "source_sha256": row["sha256"], "source_bytes": len(data), "source_commit": commit,
            "origin": origin, "archive": index.identity(archive), "archive_is_historical_evidence_not_current_executable": True})
    atomic_write_json(destination.parent / "producer_sources.json", {"status": "BYTE_EXACT_SOURCES_RESTORED", "sources": results})
    return results
