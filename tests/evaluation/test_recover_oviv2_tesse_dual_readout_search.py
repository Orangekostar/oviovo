from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.evaluation.recover_oviv2_tesse_dual_readout_search as recovery_module
from scripts.evaluation.recover_oviv2_tesse_dual_readout_search import recover_search
from scripts.evaluation.run_oviv2_tesse_dual_readout_search import (
    input_binding_values_sha256,
    non_temporal_config_sha256,
)
from scripts.evaluation.oviv2_tesse_cd_v2_config import canonical_algorithm_hash


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "configs/evaluation/manifests/oviv2_tesse_dual_readout_search_v1.json"
)
APARTMENT_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_apartment_v2.json"
OFFICE_CONFIG = REPO_ROOT / "configs/oviv2_tesse_cd_office_v2.json"
CANDIDATE_IDS = ("a0", "a1", "a2", "a3", "a4")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _content_record(path: Path) -> dict[str, Any]:
    record = _file_record(path)
    return {"sha256": record["sha256"], "byte_count": record["byte_count"]}


def _runner_input_sha256(
    config_path: Path,
    schedule_path: Path,
    target_path: Path,
    source_bindings: dict[str, Any],
) -> str:
    def byte_record(path: Path) -> dict[str, Any]:
        data = path.read_bytes()
        return {"sha256": hashlib.sha256(data).hexdigest(), "byte_count": len(data)}

    return hashlib.sha256(
        _canonical(
            {
                "config": byte_record(config_path),
                "schedule": byte_record(schedule_path),
                "target": byte_record(target_path),
                "cache_bindings": source_bindings,
            }
        )
    ).hexdigest()


def _export_digest(paths: list[Path], camera: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(camera.parent))):
        relative = str(path.relative_to(camera.parent))
        digest.update(
            relative.encode("utf-8")
            + b"\0"
            + hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii")
            + b"\n"
        )
    return digest.hexdigest()


class RecoveryFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = _write_json(
            root / "search_manifest.json", json.loads(MANIFEST.read_text())
        )
        self.manifest_payload = json.loads(self.manifest.read_text())
        self.temporal_manifest = _write_json(
            root / "temporal_frontend_manifest.json", {"temporal_only": True}
        )
        self.schedule = _write_json(root / "schedule.json", {"schedule": True})
        self.target = _write_json(root / "target.json", {"target": True})
        self.dataset_root = root / "dataset" / "apartment"
        (self.dataset_root / "results").mkdir(parents=True)
        self.rgbd_paths = []
        for index in range(2):
            for prefix, suffix in (("frame", "jpg"), ("depth", "png")):
                path = self.dataset_root / "results" / f"{prefix}{index:06d}.{suffix}"
                path.write_bytes(f"{prefix}-{index}\n".encode("ascii"))
                self.rgbd_paths.append(path)
        self.source_paths = {
            "input_manifest": _write_json(root / "input_manifest.json", {"input": True}),
            "camera": _write_json(root / "dataset" / "cam_params.json", {"camera": True}),
            "trajectory": root / "dataset" / "apartment" / "traj.txt",
            "timestamps": root / "dataset" / "apartment" / "timestamps.csv",
            "vocabulary_json": _write_json(root / "vocabulary.json", {"classes": []}),
            "vocabulary_txt": root / "vocabulary.txt",
            "frontend_manifest": _write_json(
                root / "frontend_manifest.json", {"frontend": True}
            ),
            "dense_manifest": _write_json(root / "dense_manifest.json", {"dense": True}),
        }
        self.source_paths["trajectory"].write_text("trajectory\n")
        self.source_paths["timestamps"].write_text("timestamp\n")
        self.source_paths["vocabulary_txt"].write_text("unknown\n")
        export_paths = [
            *self.rgbd_paths,
            self.source_paths["trajectory"],
            self.source_paths["timestamps"],
            self.source_paths["camera"],
        ]
        self.combined_output_sha256 = _export_digest(
            export_paths, self.source_paths["camera"]
        )
        self.source_paths["export_manifest"] = _write_json(
            self.dataset_root / "export_manifest.json",
            {
                "combined_output_sha256": self.combined_output_sha256,
                "file_hash_count": len(export_paths),
            },
        )
        apartment_payload = json.loads(APARTMENT_CONFIG.read_text())
        office_payload = json.loads(OFFICE_CONFIG.read_text())
        for payload in (apartment_payload, office_payload):
            payload["frame_count"] = 2
            payload["temporal_frontend_manifest"] = str(self.temporal_manifest)
            payload["schedule_manifest"] = str(self.schedule)
            payload["occlusion_target_manifest"] = str(self.target)
            payload["dataset_root"] = str(self.dataset_root)
            for field in (
                "input_manifest",
                "export_manifest",
                "vocabulary_json",
                "vocabulary_txt",
                "frontend_manifest",
                "dense_manifest",
            ):
                payload[field] = str(self.source_paths[field])
            payload["algorithm_hash"] = canonical_algorithm_hash(payload)
        self.apartment = _write_json(
            root / "apartment.json", apartment_payload
        )
        self.office = _write_json(
            root / "office.json", office_payload
        )
        self.apartment_payload = json.loads(self.apartment.read_text())
        self.office_payload = json.loads(self.office.read_text())
        self.code_commit = "1" * 40
        self.source_bindings: dict[str, Any] = {
            name: _content_record(path) for name, path in self.source_paths.items()
        }
        self.source_bindings[
            "rgbd_combined_output_sha256"
        ] = self.combined_output_sha256
        from tests.evaluation import (
            test_run_oviv2_tesse_dual_readout_search as search_test_helpers,
        )

        previous_manifest = search_test_helpers.MANIFEST
        previous_apartment = search_test_helpers.APARTMENT_CONFIG
        try:
            search_test_helpers.MANIFEST = self.manifest
            search_test_helpers.APARTMENT_CONFIG = self.apartment
            self.preflight = search_test_helpers._preflight(root, CANDIDATE_IDS)
        finally:
            search_test_helpers.MANIFEST = previous_manifest
            search_test_helpers.APARTMENT_CONFIG = previous_apartment
        self.preflight_record = _file_record(self.preflight)
        self.original_root = root / "original"
        self.retry_root = root / "retry"
        self.original_status = self._make_status(
            self.original_root, CANDIDATE_IDS, failed_id="a4"
        )
        self.retry_status = self._make_status(self.retry_root, ("a4",))

    def _candidate_config(self, candidate_id: str) -> dict[str, Any]:
        declaration = next(
            item
            for item in self.manifest_payload["candidates"]
            if item["candidate_id"] == candidate_id
        )
        config = copy.deepcopy(self.apartment_payload)
        config["temporal_readout"] = copy.deepcopy(declaration["temporal_readout"])
        config["algorithm_hash"] = canonical_algorithm_hash(config)
        return config

    def _candidate_record(
        self, search_root: Path, candidate_id: str, *, failed: bool
    ) -> dict[str, Any]:
        candidate_root = search_root / "candidates" / candidate_id / "apartment"
        config = self._candidate_config(candidate_id)
        config_path = _write_json(candidate_root / "config.json", config)
        stdout = candidate_root / "stdout.log"
        stderr = candidate_root / "stderr.log"
        stdout.write_bytes(b"" if failed else b"completed\n")
        stderr.write_bytes(b"")
        output_root = candidate_root / "run"
        output_root.mkdir()
        run_identity = None
        input_hashes = None
        if not failed:
            source_bindings = copy.deepcopy(self.source_bindings)
            if candidate_id not in {"a0", "a1"}:
                temporal_data = self.temporal_manifest.read_bytes()
                source_bindings["temporal_frontend_manifest"] = {
                    "sha256": hashlib.sha256(temporal_data).hexdigest(),
                    "byte_count": len(temporal_data),
                }
            run_manifest = {
                "algorithm_hash": config["algorithm_hash"],
                "input_sha256": _runner_input_sha256(
                    config_path,
                    self.schedule,
                    self.target,
                    source_bindings,
                ),
                "code_commit": self.code_commit,
                "source_bindings": source_bindings,
                "config": _file_record(config_path),
            }
            _write_json(output_root / "run_manifest.json", run_manifest)
            input_hashes = copy.deepcopy(source_bindings)
            run_identity = {
                key: copy.deepcopy(run_manifest[key])
                for key in (
                    "algorithm_hash",
                    "input_sha256",
                    "code_commit",
                    "source_bindings",
                )
            }
        config_bytes = _canonical(config)
        return {
            "candidate_id": candidate_id,
            "scene": "apartment",
            "status": "FAIL" if failed else "PASS",
            "command": ["python", "runner.py"],
            "pid": 1234,
            "exit_code": -9 if failed else 0,
            "cuda_visible_devices": "0",
            "config_path": str(config_path.absolute()),
            "config_file": _file_record(config_path),
            "output_root": str(output_root.absolute()),
            "stdout_path": str(stdout.absolute()),
            "stderr_path": str(stderr.absolute()),
            "stdout_file": _file_record(stdout),
            "stderr_file": _file_record(stderr),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "algorithm_hash": config["algorithm_hash"],
            "non_temporal_config_sha256": non_temporal_config_sha256(config),
            "input_binding_values_sha256": input_binding_values_sha256(config),
            "input_hashes": input_hashes,
            "run_identity": run_identity,
            "runtime_seconds": 1.0,
            "failure_reason": None,
        }

    def _make_status(
        self,
        search_root: Path,
        candidate_ids: tuple[str, ...],
        *,
        failed_id: str | None = None,
    ) -> Path:
        status = {
            "schema_version": 1,
            "manifest": {
                "path": str(self.manifest.absolute()),
                "sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            },
            "apartment_base_config": {
                "path": str(self.apartment.absolute()),
                "file_sha256": hashlib.sha256(self.apartment.read_bytes()).hexdigest(),
            },
            "office_binding": {
                "scene": "office",
                "executed": False,
                "config_path": str(self.office.absolute()),
                "file_sha256": hashlib.sha256(self.office.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(
                    _canonical(self.office_payload)
                ).hexdigest(),
                "algorithm_hash": self.office_payload["algorithm_hash"],
            },
            "preflight_gate_evidence": copy.deepcopy(self.preflight_record),
            "max_parallel": 1,
            "gpu_ids": ["0"],
            "required_available_ram_bytes": 1,
            "observed_available_ram_bytes": 2,
            "status": "FAIL" if failed_id else "PASS",
            "candidates": [
                self._candidate_record(
                    search_root, candidate_id, failed=candidate_id == failed_id
                )
                for candidate_id in candidate_ids
            ],
            "unscheduled_candidate_ids": [],
        }
        return _write_json(search_root / "search_status.json", status)

    def rewrite_status(self, which: str, mutate: Any) -> Path:
        path = getattr(self, f"{which}_status")
        payload = json.loads(path.read_text())
        mutate(payload)
        _write_json(path, payload)
        return path

    def rewrite_preflight(self, mutate: Any) -> None:
        payload = json.loads(self.preflight.read_text())
        mutate(payload)
        _write_json(self.preflight, payload)
        self.preflight_record = _file_record(self.preflight)
        for which in ("original", "retry"):
            self.rewrite_status(
                which,
                lambda status: status.__setitem__(
                    "preflight_gate_evidence",
                    copy.deepcopy(self.preflight_record),
                ),
            )

    def recover(self, output: Path | None = None) -> Path:
        return recover_search(
            manifest_path=self.manifest,
            original_status=self.original_status,
            retry_status=self.retry_status,
            output_root=output or self.root / "recovered",
        )


@pytest.fixture
def recovery_fixture(tmp_path: Path) -> RecoveryFixture:
    return RecoveryFixture(tmp_path)


def test_recovers_exact_a4_sigkill(recovery_fixture: RecoveryFixture) -> None:
    result = recovery_fixture.recover()

    recovered = json.loads(result.read_text())
    assert recovered["status"] == "PASS"
    assert recovered["preflight_gate_evidence"] == recovery_fixture.preflight_record


def test_rejects_distinct_preflight_records_even_when_bytes_match(
    recovery_fixture: RecoveryFixture,
) -> None:
    retry_preflight = recovery_fixture.root / "retry-preflight.json"
    retry_preflight.write_bytes(recovery_fixture.preflight.read_bytes())
    recovery_fixture.rewrite_status(
        "retry",
        lambda status: status.__setitem__(
            "preflight_gate_evidence", _file_record(retry_preflight)
        ),
    )

    with pytest.raises(ValueError, match="preflight.*record"):
        recovery_fixture.recover()


@pytest.mark.parametrize("candidate_index", range(4))
def test_rejects_preflight_failure_for_original_a0_a3(
    recovery_fixture: RecoveryFixture, candidate_index: int
) -> None:
    candidate_id = CANDIDATE_IDS[candidate_index]
    recovery_fixture.rewrite_preflight(
        lambda evidence: evidence["candidates"][candidate_index][
            "future_leakage"
        ].__setitem__("count", 1)
    )

    with pytest.raises(ValueError, match=f"preflight candidate {candidate_id}.*PASS"):
        recovery_fixture.recover()


def test_relative_temporal_manifest_is_resolved_from_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repo"
    temporal = _write_json(
        repository_root / "inputs/temporal.json", {"temporal": True}
    )
    data = temporal.read_bytes()
    binding = {
        "temporal_frontend_manifest": {
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_count": len(data),
        }
    }
    monkeypatch.setattr(recovery_module, "REPO_ROOT", repository_root)

    snapshot = recovery_module._snapshot_temporal_frontend_manifest(
        {"temporal_frontend_manifest": "inputs/temporal.json"},
        binding,
        "relative candidate",
    )

    assert snapshot is not None
    assert snapshot.path == temporal.absolute()


@pytest.mark.parametrize("candidate_index", range(4))
def test_rejects_failed_original_a0_a3_from_final_pass(
    recovery_fixture: RecoveryFixture, candidate_index: int
) -> None:
    candidate_id = CANDIDATE_IDS[candidate_index]

    def mark_failed(status: dict[str, Any]) -> None:
        record = status["candidates"][candidate_index]
        record["status"] = "FAIL"
        record["exit_code"] = 1

    recovery_fixture.rewrite_status("original", mark_failed)

    with pytest.raises(ValueError, match=f"original {candidate_id.upper()} must be PASS"):
        recovery_fixture.recover()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exit_code", 1, "exit code"),
        ("failure_reason", "out of memory", "failure reason"),
        ("input_hashes", {}, "input hashes"),
        ("run_identity", {}, "run identity"),
    ],
)
def test_rejects_invalid_original_a4_failure_fields(
    recovery_fixture: RecoveryFixture,
    field: str,
    value: object,
    message: str,
) -> None:
    recovery_fixture.rewrite_status(
        "original", lambda status: status["candidates"][4].__setitem__(field, value)
    )

    with pytest.raises(ValueError, match=message):
        recovery_fixture.recover()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_rejects_nonempty_original_a4_logs(
    recovery_fixture: RecoveryFixture, stream: str
) -> None:
    status = json.loads(recovery_fixture.original_status.read_text())
    path = Path(status["candidates"][4][f"{stream}_path"])
    path.write_bytes(b"unexpected")
    status["candidates"][4][f"{stream}_file"] = _file_record(path)
    _write_json(recovery_fixture.original_status, status)

    with pytest.raises(ValueError, match=f"{stream}.*empty"):
        recovery_fixture.recover()


def test_rejects_original_candidates_out_of_order(
    recovery_fixture: RecoveryFixture,
) -> None:
    recovery_fixture.rewrite_status(
        "original", lambda status: status["candidates"].reverse()
    )

    with pytest.raises(ValueError, match="exact A0-A4 order"):
        recovery_fixture.recover()


def test_rejects_retry_that_is_not_exactly_a4(
    recovery_fixture: RecoveryFixture,
) -> None:
    recovery_fixture.rewrite_status(
        "retry", lambda status: status["candidates"][0].__setitem__("candidate_id", "a3")
    )

    with pytest.raises(ValueError, match="exactly one PASS A4"):
        recovery_fixture.recover()


@pytest.mark.parametrize(
    ("which", "path", "message"),
    [
        ("original", (), "search status keys mismatch"),
        ("retry", ("candidates", 0), "candidate record keys mismatch"),
    ],
)
def test_rejects_unrecognized_status_or_candidate_fields(
    recovery_fixture: RecoveryFixture,
    which: str,
    path: tuple[object, ...],
    message: str,
) -> None:
    def mutate(status: dict[str, Any]) -> None:
        target: dict[str, Any] = status
        for item in path:
            target = target[item]  # type: ignore[index]
        target["unvalidated"] = True

    recovery_fixture.rewrite_status(which, mutate)

    with pytest.raises(ValueError, match=message):
        recovery_fixture.recover()


@pytest.mark.parametrize("which", ["original", "retry"])
@pytest.mark.parametrize("schema_version", [True, "1", 0])
def test_rejects_noncanonical_status_schema_version(
    recovery_fixture: RecoveryFixture, which: str, schema_version: object
) -> None:
    recovery_fixture.rewrite_status(
        which,
        lambda status: status.__setitem__("schema_version", schema_version),
    )

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        recovery_fixture.recover()


@pytest.mark.parametrize("which", ["original", "retry"])
def test_rejects_unscheduled_candidates(
    recovery_fixture: RecoveryFixture, which: str
) -> None:
    recovery_fixture.rewrite_status(
        which, lambda status: status.__setitem__("unscheduled_candidate_ids", ["a4"])
    )

    with pytest.raises(ValueError, match="unscheduled"):
        recovery_fixture.recover()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("manifest", "manifest.*mismatch"),
        ("apartment", "Apartment base config.*mismatch"),
        ("office", "Office binding.*mismatch"),
        ("a4_bytes", "A4 config bytes differ"),
        ("declared_temporal", "manifest declaration"),
        ("non_temporal", "non-temporal.*mismatch"),
        ("input_binding", "input binding values.*mismatch"),
        ("run_input", "run algorithm/input.*mismatch"),
        ("code_commit", "code commit differs"),
        ("source_bindings", "source binding"),
    ],
)
def test_rejects_cross_run_binding_mismatch(
    recovery_fixture: RecoveryFixture, scenario: str, message: str
) -> None:
    original = json.loads(recovery_fixture.original_status.read_text())
    retry = json.loads(recovery_fixture.retry_status.read_text())
    if scenario == "manifest":
        retry["manifest"]["sha256"] = "f" * 64
    elif scenario == "apartment":
        apartment = json.loads(recovery_fixture.apartment.read_text())
        apartment["block_count"] += 1
        apartment["algorithm_hash"] = canonical_algorithm_hash(apartment)
        _write_json(recovery_fixture.apartment, apartment)
        digest = hashlib.sha256(recovery_fixture.apartment.read_bytes()).hexdigest()
        original["apartment_base_config"]["file_sha256"] = digest
        retry["apartment_base_config"]["file_sha256"] = digest
    elif scenario == "office":
        office = json.loads(recovery_fixture.office.read_text())
        office["block_count"] += 1
        office["algorithm_hash"] = canonical_algorithm_hash(office)
        _write_json(recovery_fixture.office, office)
        for status in (original, retry):
            status["office_binding"]["file_sha256"] = hashlib.sha256(
                recovery_fixture.office.read_bytes()
            ).hexdigest()
            status["office_binding"]["config_sha256"] = hashlib.sha256(
                _canonical(office)
            ).hexdigest()
            status["office_binding"]["algorithm_hash"] = office["algorithm_hash"]
    elif scenario == "a4_bytes":
        config_path = Path(retry["candidates"][0]["config_path"])
        config_path.write_bytes(config_path.read_bytes() + b" ")
        retry["candidates"][0]["config_file"] = _file_record(config_path)
    elif scenario == "declared_temporal":
        record = original["candidates"][0]
        config_path = Path(record["config_path"])
        config = json.loads(config_path.read_text())
        config["temporal_readout"] = copy.deepcopy(
            recovery_fixture.manifest_payload["candidates"][1]["temporal_readout"]
        )
        config["algorithm_hash"] = canonical_algorithm_hash(config)
        _write_json(config_path, config)
        record["config_path"] = str(config_path.absolute())
        record["config_file"] = _file_record(config_path)
        record["config_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
        record["algorithm_hash"] = config["algorithm_hash"]
        run_path = Path(record["output_root"]) / "run_manifest.json"
        run = json.loads(run_path.read_text())
        run["algorithm_hash"] = config["algorithm_hash"]
        run["config"] = _file_record(config_path)
        _write_json(run_path, run)
        record["run_identity"]["algorithm_hash"] = config["algorithm_hash"]
    elif scenario == "non_temporal":
        retry["candidates"][0]["non_temporal_config_sha256"] = "f" * 64
    elif scenario == "input_binding":
        retry["candidates"][0]["input_binding_values_sha256"] = "f" * 64
    elif scenario == "run_input":
        record = retry["candidates"][0]
        run_path = Path(record["output_root"]) / "run_manifest.json"
        run = json.loads(run_path.read_text())
        run["input_sha256"] = "f" * 64
        _write_json(run_path, run)
    elif scenario == "code_commit":
        record = retry["candidates"][0]
        run_path = Path(record["output_root"]) / "run_manifest.json"
        run = json.loads(run_path.read_text())
        run["code_commit"] = "e" * 40
        record["run_identity"]["code_commit"] = "e" * 40
        _write_json(run_path, run)
    elif scenario == "source_bindings":
        record = retry["candidates"][0]
        changed = {"dense_manifest": "e" * 64, "input_manifest": "3" * 64}
        run_path = Path(record["output_root"]) / "run_manifest.json"
        run = json.loads(run_path.read_text())
        run["source_bindings"] = changed
        record["input_hashes"] = copy.deepcopy(changed)
        record["run_identity"]["source_bindings"] = copy.deepcopy(changed)
        _write_json(run_path, run)
    _write_json(recovery_fixture.original_status, original)
    _write_json(recovery_fixture.retry_status, retry)

    with pytest.raises(ValueError, match=message):
        recovery_fixture.recover()


def test_rejects_coordinated_input_sha256_tampering_across_all_pass_runs(
    recovery_fixture: RecoveryFixture,
) -> None:
    for which in ("original", "retry"):
        status = json.loads(getattr(recovery_fixture, f"{which}_status").read_text())
        for record in status["candidates"]:
            if record["status"] != "PASS":
                continue
            run_path = Path(record["output_root"]) / "run_manifest.json"
            run = json.loads(run_path.read_text())
            run["input_sha256"] = "f" * 64
            record["run_identity"]["input_sha256"] = "f" * 64
            _write_json(run_path, run)
        _write_json(getattr(recovery_fixture, f"{which}_status"), status)

    with pytest.raises(ValueError, match="input_sha256"):
        recovery_fixture.recover()


def test_rejects_coordinated_dense_manifest_binding_tampering_across_all_pass_runs(
    recovery_fixture: RecoveryFixture,
) -> None:
    for which in ("original", "retry"):
        status = json.loads(getattr(recovery_fixture, f"{which}_status").read_text())
        for record in status["candidates"]:
            if record["status"] != "PASS":
                continue
            run_path = Path(record["output_root"]) / "run_manifest.json"
            run = json.loads(run_path.read_text())
            run["source_bindings"]["dense_manifest"]["sha256"] = "f" * 64
            run["input_sha256"] = _runner_input_sha256(
                Path(record["config_path"]),
                recovery_fixture.schedule,
                recovery_fixture.target,
                run["source_bindings"],
            )
            record["input_hashes"] = copy.deepcopy(run["source_bindings"])
            record["run_identity"]["source_bindings"] = copy.deepcopy(
                run["source_bindings"]
            )
            record["run_identity"]["input_sha256"] = run["input_sha256"]
            _write_json(run_path, run)
        _write_json(getattr(recovery_fixture, f"{which}_status"), status)

    with pytest.raises(ValueError, match="dense_manifest.*source binding"):
        recovery_fixture.recover()


def test_rejects_coordinated_camera_binding_tampering_across_all_pass_runs(
    recovery_fixture: RecoveryFixture,
) -> None:
    camera = recovery_fixture.source_paths["camera"]
    camera.write_bytes(b'{"camera":"tampered"}\n')
    changed_binding = _content_record(camera)
    for which in ("original", "retry"):
        status = json.loads(getattr(recovery_fixture, f"{which}_status").read_text())
        for record in status["candidates"]:
            if record["status"] != "PASS":
                continue
            run_path = Path(record["output_root"]) / "run_manifest.json"
            run = json.loads(run_path.read_text())
            run["source_bindings"]["camera"] = copy.deepcopy(changed_binding)
            run["input_sha256"] = _runner_input_sha256(
                Path(record["config_path"]),
                recovery_fixture.schedule,
                recovery_fixture.target,
                run["source_bindings"],
            )
            record["input_hashes"] = copy.deepcopy(run["source_bindings"])
            record["run_identity"]["source_bindings"] = copy.deepcopy(
                run["source_bindings"]
            )
            record["run_identity"]["input_sha256"] = run["input_sha256"]
            _write_json(run_path, run)
        _write_json(getattr(recovery_fixture, f"{which}_status"), status)

    with pytest.raises(ValueError, match="combined output"):
        recovery_fixture.recover()


def test_publication_builds_composite_tree_and_provenance(
    recovery_fixture: RecoveryFixture,
) -> None:
    original_bytes = recovery_fixture.original_status.read_bytes()
    retry_bytes = recovery_fixture.retry_status.read_bytes()
    original = json.loads(original_bytes)
    retry = json.loads(retry_bytes)

    result = recovery_fixture.recover()
    status = json.loads(result.read_text())

    assert status["candidates"][:4] == original["candidates"][:4]
    assert status["candidates"][4] == retry["candidates"][0]
    candidate_root = result.parent / "candidates"
    assert sorted(path.parent.name for path in candidate_root.glob("*/apartment")) == list(
        CANDIDATE_IDS
    )
    recovery = status["recovery"]
    assert recovery["strategy"] == "immutable_single_candidate_retry_v1"
    assert recovery["replaced_candidate_id"] == "a4"
    assert recovery["original_status"]["sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    assert recovery["retry_status"]["sha256"] == hashlib.sha256(retry_bytes).hexdigest()
    assert recovery["accepted_original_failure"]["exit_code"] == -9
    assert recovery["accepted_original_failure"]["failure_reason"] is None


def test_publication_no_clobber_preserves_existing_output(
    recovery_fixture: RecoveryFixture,
) -> None:
    output = recovery_fixture.root / "existing"
    output.mkdir()
    marker = output / "marker"
    marker.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        recovery_fixture.recover(output)

    assert marker.read_bytes() == b"keep"


def test_publication_rejects_source_mutation_before_publish(
    recovery_fixture: RecoveryFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = recovery_fixture.root / "mutated-output"
    original_revalidate = recovery_module._revalidate_sources

    def mutate_then_revalidate(witnesses: list[object]) -> None:
        recovery_fixture.original_status.write_bytes(
            recovery_fixture.original_status.read_bytes() + b" "
        )
        original_revalidate(witnesses)

    monkeypatch.setattr(recovery_module, "_revalidate_sources", mutate_then_revalidate)

    with pytest.raises(ValueError, match="changed after snapshot"):
        recovery_fixture.recover(output)

    assert not output.exists()


def test_publication_rejects_temporal_manifest_mutation_before_publish(
    recovery_fixture: RecoveryFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = recovery_fixture.root / "mutated-temporal-output"
    original_revalidate = recovery_module._revalidate_sources

    def mutate_then_revalidate(witnesses: list[object]) -> None:
        recovery_fixture.temporal_manifest.write_bytes(b'{"changed":true}\n')
        original_revalidate(witnesses)

    monkeypatch.setattr(recovery_module, "_revalidate_sources", mutate_then_revalidate)

    with pytest.raises(ValueError, match="changed after snapshot"):
        recovery_fixture.recover(output)

    assert not output.exists()


def test_publication_rejects_preflight_mutation_before_publish(
    recovery_fixture: RecoveryFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = recovery_fixture.root / "mutated-preflight-output"
    original_revalidate = recovery_module._revalidate_sources

    def mutate_then_revalidate(witnesses: list[object]) -> None:
        recovery_fixture.preflight.write_bytes(
            recovery_fixture.preflight.read_bytes() + b" "
        )
        original_revalidate(witnesses)

    monkeypatch.setattr(recovery_module, "_revalidate_sources", mutate_then_revalidate)

    with pytest.raises(ValueError, match="changed after snapshot"):
        recovery_fixture.recover(output)

    assert not output.exists()


def test_publication_rename_failure_cleans_staging_and_reservation(
    recovery_fixture: RecoveryFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = recovery_fixture.root / "rename-output"
    real_rename = recovery_module.os.rename

    def fail_staging_rename(source: object, destination: object) -> None:
        if ".rename-output.staging-" in str(source):
            raise OSError("injected rename failure")
        real_rename(source, destination)

    monkeypatch.setattr(recovery_module.os, "rename", fail_staging_rename)

    with pytest.raises(OSError, match="injected rename failure"):
        recovery_fixture.recover(output)

    assert not output.exists()
    assert not list(recovery_fixture.root.glob(".rename-output.staging-*"))
