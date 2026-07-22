from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import stat

import pytest

import scripts.evaluation.derive_tesse_cd_causal_schedule as causal_schedule
from scripts.evaluation.derive_tesse_cd_causal_schedule import (
    build_schedule,
    derive_scene_schedule,
    load_depth_timestamps,
    load_event_timestamps,
    render_schedule,
    run,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/evaluation/manifests/tesse_cd.json"
FROZEN_SCHEDULE = (
    ROOT / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_database(path: Path, timestamps: list[int]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("create table topics(id integer primary key, name text)")
        connection.execute(
            "create table messages("
            "id integer primary key, topic_id integer not null, "
            "timestamp integer not null, data blob)"
        )
        connection.execute("insert into topics values(7, '/depth')")
        connection.executemany(
            "insert into messages values(?, 7, ?, X'')",
            [(index + 1, timestamp) for index, timestamp in enumerate(timestamps)],
        )


def _write_changes(path: Path, event_times: list[int], terminal: int | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("ObjectSymbol", "AppearedAt", "DisappearedAt"))
        for index, event_time in enumerate(event_times):
            writer.writerow((f"O({index})", 0, event_time))
        if terminal is not None:
            writer.writerow(("O(terminal)", 0, terminal))


def _declared_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_manifest(
    root: Path,
    *,
    timestamps: list[int],
    event_times: list[int],
) -> Path:
    sequences: dict[str, object] = {}
    for scene in ("apartment", "office"):
        database = root / f"{scene}.db3"
        changes = root / f"{scene}_changes.csv"
        _write_database(database, timestamps)
        _write_changes(changes, event_times, terminal=timestamps[-1] - timestamps[0] + 1)
        sequences[scene] = {
            "bag": {"database": _declared_file(database)},
            "timeline": {
                "depth_frame_count": len(timestamps),
                "first_depth_timestamp_ns": timestamps[0],
                "last_depth_timestamp_ns": timestamps[-1],
                "change_times_relative_ns": event_times,
            },
            "ground_truth": {"files": {"changes": _declared_file(changes)}},
        }
    manifest = root / "tesse.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "tesse_cd_dynamic_v1",
                "dataset": "TESSE-CD",
                "source": {"owner": "MIT-SPARK Khronos official release"},
                "topics": {"depth": "/depth"},
                "sequences": sequences,
                "protocol": {
                    "future_frames_allowed": False,
                    "ground_truth_evaluator_only": True,
                    "runtime_ground_truth_access": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_real_assets_derive_interventions_official_stride_and_common_horizons() -> None:
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_interventions = {
        "apartment": [263, 711, 858, 1022],
        "office": [2001, 2401, 2601, 3601],
    }
    expected_official = {
        "apartment": [450, 900, 1350],
        "office": [450, 900, 1350, 1800, 2250, 2700, 3150, 3600, 4050],
    }

    for scene in ("apartment", "office"):
        sequence = source["sequences"][scene]
        timestamps = load_depth_timestamps(
            Path(sequence["bag"]["database"]["path"]),
            topic=source["topics"]["depth"],
        )
        event_times = load_event_timestamps(
            Path(sequence["ground_truth"]["files"]["changes"]["path"]),
            last_relative_timestamp_ns=timestamps[-1] - timestamps[0],
        )
        schedule = derive_scene_schedule(
            scene,
            timestamps,
            event_times,
            official_stride=450,
            common_step=50,
            common_horizon=450,
        )

        assert [event["intervention_frame_index"] for event in schedule["events"]] == (
            expected_interventions[scene]
        )
        assert [
            entry["frame_index"]
            for entry in schedule["entries"]
            if "official" in entry["roles"]
        ] == expected_official[scene]
        for event in schedule["events"]:
            intervention = event["intervention_frame_index"]
            assert [
                entry["frame_index"]
                for entry in schedule["entries"]
                if event["event_id"] in entry["event_ids"]
            ] == list(range(intervention, intervention + 451, 50))
        assert all(
            set(entry) == {
                "frame_index",
                "timestamp_ns",
                "relative_timestamp_ns",
                "event_ids",
                "roles",
            }
            for entry in schedule["entries"]
        )


def test_build_schedule_hash_binds_sources_and_is_byte_deterministic(tmp_path: Path) -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]
    manifest = _write_manifest(tmp_path, timestamps=timestamps, event_times=[250])

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    run(
        manifest,
        first,
        official_stride=5,
        common_step=2,
        common_horizon=18,
    )
    run(
        manifest,
        second,
        official_stride=5,
        common_step=2,
        common_horizon=18,
    )

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["manifest_id"] == "tesse_cd_causal_schedule_v2"
    assert payload["parameters"] == {
        "frame_indexing": "zero_based",
        "event_frame_rule": "first relative_timestamp_ns >= event timestamp",
        "official_stride_frames": 5,
        "common_event_step_frames": 2,
        "common_event_horizon_frames": 18,
        "common_checkpoints_per_event": 10,
    }
    for scene in ("apartment", "office"):
        sources = payload["scenes"][scene]["sources"]
        assert set(sources) == {"database", "gt_changes"}
        for entry in sources.values():
            path = Path(entry["path"])
            assert entry["sha256"] == _sha256(path)
            assert entry["byte_count"] == path.stat().st_size


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("sha256", "0" * 64, "database SHA256 mismatch"),
        ("size_bytes", 1, "database byte count mismatch"),
    ],
)
def test_build_schedule_rejects_database_provenance_mismatch(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]
    manifest_path = _write_manifest(tmp_path, timestamps=timestamps, event_times=[250])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sequences"]["apartment"]["bag"]["database"][field] = bad_value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_schedule(
            manifest_path,
            official_stride=5,
            common_step=2,
            common_horizon=18,
        )


def test_build_schedule_rejects_depth_frame_count_mismatch(tmp_path: Path) -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]
    manifest_path = _write_manifest(tmp_path, timestamps=timestamps, event_times=[250])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sequences"]["office"]["timeline"]["depth_frame_count"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="depth frame count mismatch"):
        build_schedule(
            manifest_path,
            official_stride=5,
            common_step=2,
            common_horizon=18,
        )


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ([100, 150, 150], "duplicate depth timestamp"),
        ([100, 200, 150], "non-monotonic depth timestamp"),
    ],
)
def test_load_depth_timestamps_rejects_duplicate_or_nonmonotonic_frames(
    tmp_path: Path, timestamps: list[int], message: str
) -> None:
    database = tmp_path / "bag.db3"
    _write_database(database, timestamps)

    with pytest.raises(ValueError, match=message):
        load_depth_timestamps(database, topic="/depth")


def test_derive_scene_schedule_rejects_incomplete_event_horizon() -> None:
    timestamps = [1000 + 10 * index for index in range(12)]

    with pytest.raises(ValueError, match="ten in-horizon checkpoints"):
        derive_scene_schedule(
            "apartment",
            timestamps,
            [80],
            official_stride=5,
            common_step=2,
            common_horizon=18,
        )


def test_run_rejects_output_reuse(tmp_path: Path) -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]
    manifest = _write_manifest(tmp_path, timestamps=timestamps, event_times=[250])
    output = tmp_path / "schedule.json"
    output.write_text("occupied\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output already exists"):
        run(
            manifest,
            output,
            official_stride=5,
            common_step=2,
            common_horizon=18,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        (("schema_version",), 2),
        (("manifest_id",), "lookalike"),
        (("source", "owner"), "lookalike owner"),
        (("protocol", "future_frames_allowed"), True),
        (("protocol", "ground_truth_evaluator_only"), False),
        (("protocol", "runtime_ground_truth_access"), True),
    ],
)
def test_build_schedule_rejects_noncanonical_source_identity_by_default(
    tmp_path: Path, field: tuple[str, ...], bad_value: object
) -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]
    manifest_path = _write_manifest(tmp_path, timestamps=timestamps, event_times=[250])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["manifest_id"] = "tesse_cd_dynamic_v1"
    target = payload
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = bad_value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical source identity"):
        build_schedule(
            manifest_path,
            official_stride=5,
            common_step=2,
            common_horizon=18,
        )


def test_public_build_and_run_do_not_expose_source_identity_bypasses() -> None:
    for function in (build_schedule, run):
        parameters = inspect.signature(function).parameters
        assert "expected_manifest_id" not in parameters
        assert "strict_source_identity" not in parameters


def test_checked_in_schedule_matches_real_derivation_and_canonical_sources() -> None:
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_SCHEDULE.read_text(encoding="utf-8"))
    assert render_schedule(build_schedule(MANIFEST)) == FROZEN_SCHEDULE.read_bytes()

    assert {
        key: frozen[key]
        for key in (
            "schema_version",
            "manifest_id",
            "dataset",
            "parameters",
            "method_predictions_used",
        )
    } == {
        "schema_version": 2,
        "manifest_id": "tesse_cd_causal_schedule_v2",
        "dataset": "TESSE-CD",
        "parameters": {
            "frame_indexing": "zero_based",
            "event_frame_rule": "first relative_timestamp_ns >= event timestamp",
            "official_stride_frames": 450,
            "common_event_step_frames": 50,
            "common_event_horizon_frames": 450,
            "common_checkpoints_per_event": 10,
        },
        "method_predictions_used": False,
    }
    assert frozen["source_manifest"] == {
        "path": MANIFEST.relative_to(ROOT).as_posix(),
        "sha256": _sha256(MANIFEST),
        "byte_count": MANIFEST.stat().st_size,
    }
    for scene in ("apartment", "office"):
        sequence = source["sequences"][scene]
        timestamps = load_depth_timestamps(
            Path(sequence["bag"]["database"]["path"]),
            topic=source["topics"]["depth"],
        )
        events = load_event_timestamps(
            Path(sequence["ground_truth"]["files"]["changes"]["path"]),
            last_relative_timestamp_ns=timestamps[-1] - timestamps[0],
        )
        derived = derive_scene_schedule(scene, timestamps, events)
        stored = frozen["scenes"][scene]
        assert {
            key: stored[key]
            for key in (
                "frame_count",
                "first_depth_timestamp_ns",
                "last_depth_timestamp_ns",
                "events",
                "entries",
            )
        } == derived
        for source_name, declaration in (
            ("database", sequence["bag"]["database"]),
            ("gt_changes", sequence["ground_truth"]["files"]["changes"]),
        ):
            assert stored["sources"][source_name] == {
                "path": str(Path(declaration["path"]).resolve()),
                "sha256": declaration["sha256"],
                "byte_count": declaration["size_bytes"],
            }


def test_load_event_timestamps_ignores_disappearance_eos_sentinel(
    tmp_path: Path,
) -> None:
    changes = tmp_path / "changes.csv"
    _write_changes(changes, [20], terminal=101)

    assert load_event_timestamps(
        changes, last_relative_timestamp_ns=100
    ) == [20]


def test_load_event_timestamps_rejects_appearance_after_sensor_stream(
    tmp_path: Path,
) -> None:
    changes = tmp_path / "changes.csv"
    with changes.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("ObjectSymbol", "AppearedAt", "DisappearedAt"))
        writer.writerow(("O(1)", 101, 0))

    with pytest.raises(ValueError, match="appearance exceeds last depth frame"):
        load_event_timestamps(changes, last_relative_timestamp_ns=100)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("", 0, 20), "ObjectSymbol must be non-empty"),
        (("O(1)", -1, 20), "must be non-negative"),
        (("O(1)", 0, 0), "lifecycle cannot be all zero"),
    ],
)
def test_load_event_timestamps_rejects_invalid_lifecycle_rows(
    tmp_path: Path, row: tuple[object, object, object], message: str
) -> None:
    changes = tmp_path / "changes.csv"
    with changes.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("ObjectSymbol", "AppearedAt", "DisappearedAt"))
        writer.writerow(row)
        writer.writerow(("O(valid)", 0, 20))

    with pytest.raises(ValueError, match=message):
        load_event_timestamps(changes, last_relative_timestamp_ns=100)


@pytest.mark.parametrize(("appeared", "disappeared"), [(10, 10), (20, 10)])
def test_load_event_timestamps_rejects_nonpositive_lifecycle_duration(
    tmp_path: Path, appeared: int, disappeared: int
) -> None:
    changes = tmp_path / "changes.csv"
    with changes.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("ObjectSymbol", "AppearedAt", "DisappearedAt"))
        writer.writerow(("O(1)", appeared, disappeared))

    with pytest.raises(
        ValueError, match="DisappearedAt must be greater than AppearedAt"
    ):
        load_event_timestamps(changes, last_relative_timestamp_ns=100)


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_declared_byte_count_accepts_only_nonnegative_plain_integers(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        causal_schedule._declared_byte_count({"size_bytes": value})


def test_derive_scene_schedule_rejects_empty_official_schedule() -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]

    with pytest.raises(ValueError, match="official checkpoint schedule is empty"):
        derive_scene_schedule(
            "apartment",
            timestamps,
            [250],
            official_stride=30,
            common_step=2,
            common_horizon=18,
        )


def test_load_depth_timestamps_uses_immutable_read_only_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "bag.db3"
    _write_database(database, [100, 150, 200])
    observed: list[str] = []
    original_connect = sqlite3.connect

    def recording_connect(target: str, *args: object, **kwargs: object):
        observed.append(target)
        return original_connect(target, *args, **kwargs)

    monkeypatch.setattr(causal_schedule.sqlite3, "connect", recording_connect)

    assert load_depth_timestamps(database, topic="/depth") == [100, 150, 200]
    assert observed == [f"file:{database}?mode=ro&immutable=1"]


def test_build_schedule_rejects_source_changed_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]
    manifest = _write_manifest(tmp_path, timestamps=timestamps, event_times=[250])
    original_loader = causal_schedule.load_depth_timestamps
    mutated = False

    def mutating_loader(database: Path, *, topic: str) -> list[int]:
        nonlocal mutated
        values = original_loader(database, topic=topic)
        if not mutated and database.name == "apartment.db3":
            stat = database.stat()
            os.utime(
                database,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            mutated = True
        return values

    monkeypatch.setattr(causal_schedule, "load_depth_timestamps", mutating_loader)

    with pytest.raises(ValueError, match="source changed while reading"):
        build_schedule(
            manifest,
            official_stride=5,
            common_step=2,
            common_horizon=18,
        )


def test_build_schedule_rejects_gt_changes_hash_mismatch(tmp_path: Path) -> None:
    timestamps = [10_000 + 50 * index for index in range(30)]
    manifest = _write_manifest(tmp_path, timestamps=timestamps, event_times=[250])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sequences"]["apartment"]["ground_truth"]["files"]["changes"][
        "sha256"
    ] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gt_changes SHA256 mismatch"):
        build_schedule(
            manifest,
            official_stride=5,
            common_step=2,
            common_horizon=18,
        )


def test_overlapping_roles_and_event_ids_have_canonical_order() -> None:
    timestamps = [10_000 + 10 * index for index in range(40)]

    schedule = derive_scene_schedule(
        "office",
        timestamps,
        [50, 90],
        official_stride=5,
        common_step=2,
        common_horizon=18,
    )

    by_frame = {entry["frame_index"]: entry for entry in schedule["entries"]}
    assert by_frame[5]["roles"] == ["official", "common_v2"]
    assert by_frame[9]["event_ids"] == ["office_event_01", "office_event_02"]


def test_run_exclusive_create_does_not_overwrite_racing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "schedule.json"

    def racing_build(*args: object, **kwargs: object) -> dict[str, object]:
        output.write_text("competitor\n", encoding="utf-8")
        return {"schema_version": 2}

    monkeypatch.setattr(causal_schedule, "build_schedule", racing_build)

    with pytest.raises(FileExistsError):
        run(tmp_path / "unused.json", output)
    assert output.read_text(encoding="utf-8") == "competitor\n"


def test_run_publishes_via_same_directory_link_and_fsyncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "schedule.json"
    links: list[tuple[Path, Path]] = []
    synced_modes: list[int] = []
    original_link = os.link
    original_fsync = os.fsync

    monkeypatch.setattr(
        causal_schedule,
        "build_schedule",
        lambda *args, **kwargs: {"schema_version": 2},
    )

    def recording_link(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        links.append((Path(source), Path(target)))
        original_link(source, target)

    def recording_fsync(descriptor: int) -> None:
        synced_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(causal_schedule.os, "link", recording_link)
    monkeypatch.setattr(causal_schedule.os, "fsync", recording_fsync)

    run(tmp_path / "unused.json", output)

    assert len(links) == 1
    temporary, published = links[0]
    assert temporary.parent == output.parent
    assert temporary != output
    assert published == output
    assert any(stat.S_ISREG(mode) for mode in synced_modes)
    assert any(stat.S_ISDIR(mode) for mode in synced_modes)
    assert list(tmp_path.iterdir()) == [output]


def test_run_interrupted_write_leaves_no_final_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "schedule.json"
    original_fdopen = os.fdopen

    monkeypatch.setattr(
        causal_schedule,
        "build_schedule",
        lambda *args, **kwargs: {"schema_version": 2},
    )

    class InterruptedWriter:
        def __init__(self, descriptor: int, mode: str) -> None:
            self.handle = original_fdopen(descriptor, mode)

        def __enter__(self) -> InterruptedWriter:
            self.handle.__enter__()
            return self

        def __exit__(self, *args: object) -> bool | None:
            return self.handle.__exit__(*args)

        def write(self, content: bytes) -> None:
            self.handle.write(content[:1])
            self.handle.flush()
            raise OSError("injected write interruption")

    monkeypatch.setattr(causal_schedule.os, "fdopen", InterruptedWriter)

    with pytest.raises(OSError, match="injected write interruption"):
        run(tmp_path / "unused.json", output)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
