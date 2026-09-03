from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from src.evaluation.tesse_oracle_inputs import (
    OracleInputError,
    build_oracle_input_manifest,
)


def _oracle_fixture(tmp_path: Path) -> Path:
    database = tmp_path / "scene.db3"
    connection = sqlite3.connect(database)
    connection.execute(
        "create table topics(id integer primary key, name text, type text, "
        "serialization_format text, offered_qos_profiles text)"
    )
    connection.execute(
        "create table messages(id integer primary key, topic_id integer, "
        "timestamp integer, data blob)"
    )
    topics = ("rgb", "depth", "semantics")
    for topic_id, name in enumerate(topics, start=1):
        connection.execute(
            "insert into topics values (?, ?, ?, ?, ?)",
            (topic_id, f"/{name}", "sensor_msgs/msg/Image", "cdr", ""),
        )
        for timestamp in (10, 20):
            connection.execute(
                "insert into messages(topic_id, timestamp, data) values (?, ?, ?)",
                (topic_id, timestamp, b"frame"),
            )
    connection.commit()
    connection.close()
    database_sha256 = hashlib.sha256(database.read_bytes()).hexdigest()
    descriptor = tmp_path / "oracle_source.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": "fixture",
                "source_database": {
                    "path": str(database),
                    "sha256": database_sha256,
                    "byte_count": database.stat().st_size,
                },
                "topics": {
                    "rgb": "/rgb",
                    "depth": "/depth",
                    "semantics": "/semantics",
                },
                "semantic_encoding": "16UC1",
                "max_timestamp_delta_ns": 0,
            }
        ),
        encoding="utf-8",
    )
    return descriptor


def test_oracle_inputs_are_diagnostic_and_source_bound(tmp_path: Path) -> None:
    output = tmp_path / "oracle.json"
    manifest = build_oracle_input_manifest(
        _oracle_fixture(tmp_path), output=output
    )

    assert manifest["condition"] == "ORACLE_DIAGNOSTIC"
    assert manifest["ranking_eligible"] is False
    assert manifest["source_bindings"]
    assert manifest["frame_count"] == 2
    assert output.exists()


def test_oracle_input_builder_fails_closed_on_unaligned_stream(tmp_path: Path) -> None:
    descriptor = _oracle_fixture(tmp_path)
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    database = Path(payload["source_database"]["path"])
    connection = sqlite3.connect(database)
    connection.execute(
        "update messages set timestamp=21 where topic_id=3 and timestamp=20"
    )
    connection.commit()
    connection.close()
    payload["source_database"]["sha256"] = hashlib.sha256(
        database.read_bytes()
    ).hexdigest()
    payload["source_database"]["byte_count"] = database.stat().st_size
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OracleInputError, match="timestamp alignment"):
        build_oracle_input_manifest(descriptor, output=tmp_path / "oracle.json")
