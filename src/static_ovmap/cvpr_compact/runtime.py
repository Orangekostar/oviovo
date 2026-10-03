"""New-task execution gates, independent of historical experiment guards."""

import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import time

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .protocol import experiment_matrix, load_spec


FREEZE_FILE = "artifacts/static_ovmap/cvpr_compact_tables_v1/freeze/experiment.json"


class LiveLeafError(RuntimeError):
    """A still-running process must be observed, never spent as a retry."""


def process_state(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (path / "cmdline").read_bytes().decode().split("\0")[:-1]
        cwd = os.readlink(path / "cwd")
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {"pid": int(pid), "start_ticks": fields[19], "argv": argv, "cwd": cwd,
            "live": fields[0] not in ("Z", "X")}


def assert_leaf_idle(directory, *, argv=None, cwd=None):
    for path in Path(directory).glob("command_*.json"):
        for row in read(path).get("attempts", []):
            if argv is not None and (row["argv"] != list(map(str, argv)) or row["cwd"] != str(Path(cwd).resolve())):
                continue
            actual = process_state(row["child_pid"]) if row.get("child_pid") else None
            if (actual and actual["live"] and actual["argv"] == row["argv"]
                    and actual["cwd"] == str(Path(row["cwd"]).resolve())
                    and (not row.get("child_start_ticks") or actual["start_ticks"] == row["child_start_ticks"])):
                raise LiveLeafError("task child is still live; observe it before resuming: " + str(actual["pid"]))
            actual = process_state(row["controller_pid"]) if row.get("controller_pid") else None
            if (row.get("status") == "RUNNING" and actual and actual["live"]
                    and row.get("controller_start_ticks") == actual["start_ticks"]
                    and row.get("controller_argv") == actual["argv"]):
                raise LiveLeafError("task controller is still live; do not start an overlapping leaf")


def require_frozen_execution(binding, scene):
    if canonical_digest({k: v for k, v in binding.items() if k != "identity"}) != binding["identity"]:
        raise ValueError("compact-task binding content changed")
    spec = load_spec(binding["spec"])
    if scene == spec["smoke_scene"]:
        if binding["scenes"][scene]["availability"] != "REUSABLE_VERIFIED_BB00_NATIVE_ANCHOR":
            raise ValueError("development smoke must reuse its verified existing anchor")
        return "DEVELOPMENT_ONLY_EXISTING_ANCHOR"
    main = [*spec["cohorts"]["replica8"], *spec["cohorts"]["scannet_cf18"]]
    if scene not in main:
        raise ValueError("execution leaves the fixed main cohorts and single allowed smoke")
    repo = Path(binding["repository_root"])
    path = repo / FREEZE_FILE
    if not path.is_file():
        raise RuntimeError("complete implementation freeze must be committed before new main predictions")
    frozen = read(path)
    if (canonical_digest({k: v for k, v in frozen.items() if k != "identity"}) != frozen["identity"]
            or frozen["status"] != "IMPLEMENTATION_FROZEN" or frozen["task_id"] != spec["task_id"]
            or frozen["binding_identity"] != binding["identity"]
            or frozen["spec_sha256"] != binding["spec_identity"]["sha256"]
            or frozen["matrix_identity"] != experiment_matrix(spec)["identity"]
            or frozen["primary_method"] != "CT_A3_ER"):
        raise ValueError("implementation freeze differs from the fixed bound experiment")
    committed = subprocess.run(["git", "show", "HEAD:" + FREEZE_FILE], cwd=repo, capture_output=True, check=False)
    if committed.returncode or committed.stdout != path.read_bytes():
        raise RuntimeError("implementation freeze is not committed at the current branch HEAD")
    revision = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", FREEZE_FILE], cwd=repo, text=True).strip()
    index = ConsumptionIndex(Path(binding["output_root"]) / "validation/input_verifications.json")
    for item in frozen["implementation_sources"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("frozen implementation source must be repository-relative")
        index.identity(repo / relative, {"sha256": item["sha256"], "bytes": item["bytes"]})
        original = subprocess.check_output(["git", "show", revision + ":" + relative.as_posix()], cwd=repo)
        if hashlib.sha256(original).hexdigest() != item["sha256"]:
            raise RuntimeError("implementation source was not committed with its experiment freeze")
    if not frozen["implementation_sources"]:
        raise ValueError("implementation freeze cannot omit its source inventory")
    return revision


def execute_leaf(argv, cwd, log_path, *, env, input_identity, prepare=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The child inherits this FD so an orphan retains its lock until it exits.
    command_key = canonical_digest({"argv": list(map(str, argv)), "cwd": str(Path(cwd).resolve())})
    with (log_path.parent / (".leaf_" + command_key + ".lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LiveLeafError("execution leaf lock is held: " + str(log_path.parent)) from exc
        assert_leaf_idle(log_path.parent, argv=argv, cwd=cwd)
        return _execute_locked(argv, cwd, log_path, env=env, input_identity=input_identity,
                               lock_fd=lock.fileno(), prepare=prepare)


def _execute_locked(argv, cwd, log_path, *, env, input_identity, lock_fd, prepare):
    ledger_path = log_path.with_name("command_" + input_identity + ".json")
    ledger = read(ledger_path) if ledger_path.is_file() else {"input_identity": input_identity, "attempts": []}
    if ledger["input_identity"] != input_identity or len(ledger["attempts"]) >= 3:
        raise RuntimeError("the identical execution leaf exhausted its two allowed retries")
    if ledger["attempts"] and ledger["attempts"][-1]["status"] == "COMPLETE":
        raise RuntimeError("successful leaf must reuse its verified result, never execute again")
    if prepare is not None:
        prepare()
    attempt = len(ledger["attempts"]) + 1
    actual_log = log_path.with_name(f"{log_path.stem}.attempt_{attempt:03d}{log_path.suffix}")
    controller = process_state(os.getpid())
    row = {"attempt": attempt, "status": "RUNNING", "argv": list(map(str, argv)), "cwd": str(Path(cwd).resolve()),
        "log": str(actual_log), "started_at_unix": time.time(), "controller_pid": os.getpid(),
        "controller_start_ticks": controller["start_ticks"], "controller_argv": controller["argv"],
        "environment": {key: value for key, value in env.items() if key.startswith("OVIMAP_") or key in
            ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH", "LD_LIBRARY_PATH")}}
    ledger["attempts"].append(row)
    atomic_write_json(ledger_path, ledger)
    started = time.monotonic()
    process = None
    try:
        with actual_log.open("wb") as stream:
            process = subprocess.Popen(row["argv"], cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
            row["child_pid"] = process.pid
            child = process_state(process.pid)
            row["child_start_ticks"] = child["start_ticks"] if child else None
            atomic_write_json(ledger_path, ledger)
            code = process.wait()
        row.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code)
    except BaseException as exc:
        row.update(status="RUNNING" if process is not None and process.poll() is None else "FAILED",
                   error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        row["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(ledger_path, ledger)
    if code:
        raise RuntimeError(f"command exited {code}; inspect {actual_log}")
    return row
