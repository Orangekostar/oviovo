from __future__ import annotations

import ast
import gzip
import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

import scripts.build_oviv2_tesse_temporal_frontend_cache as builder

CANONICAL_CLASSES = (
    "Fridge",
    "Books",
    "Chair",
    "Vase",
    "Couch",
    "Drawer",
    "Objects",
    "Table",
    "Bin",
    "Humans",
)
ALIAS_CLASSES = ("book", "cabinet", "vase", "chair")
ALIAS_MAP = {
    "book": "Books",
    "cabinet": "Drawer",
    "vase": "Vase",
    "chair": "Chair",
}
OFFICE_CLASSES = (
    "Small office objects",
    "Large static wall furniture",
    "Large office objects",
    "Bathroom",
    "Bedroom",
    "Chairs",
    "Signs",
)
OFFICE_ALIAS_CLASSES = (
    "bookshelf",
    "bookcase",
    "desk",
    "office desk",
    "table",
    "filing cabinet",
    "box",
    "cardboard box",
    "storage box",
    "cart",
    "utility cart",
    "cooler",
    "chair",
    "armchair",
    "office chair",
    "lounge chair",
    "book",
    "stack of books",
    "computer monitor",
    "toilet",
    "bed",
    "sign",
    "exit sign",
)
OFFICE_ALIAS_MAP = {
    "bookshelf": "Large static wall furniture",
    "bookcase": "Large static wall furniture",
    "desk": "Large static wall furniture",
    "office desk": "Large static wall furniture",
    "table": "Large static wall furniture",
    "filing cabinet": "Large office objects",
    "box": "Large office objects",
    "cardboard box": "Large office objects",
    "storage box": "Large office objects",
    "cart": "Large office objects",
    "utility cart": "Large office objects",
    "cooler": "Small office objects",
    "chair": "Chairs",
    "armchair": "Chairs",
    "office chair": "Chairs",
    "lounge chair": "Chairs",
    "book": "Small office objects",
    "stack of books": "Small office objects",
    "computer monitor": "Small office objects",
    "toilet": "Bathroom",
    "bed": "Bedroom",
    "sign": "Signs",
    "exit sign": "Signs",
}
IMAGE_SHAPE = (8, 8)
FEATURE_DIMENSION = 4


def test_script_uses_only_python310_typing_names() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 10))
    imported_typing_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "typing"
        for alias in node.names
    }

    assert "Self" not in imported_typing_names


def test_accepts_exact_office_canonical_schema(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    apartment = builder._load_source_manifest(fixture.canonical, "canonical")
    office = replace(apartment, scene="office", classes=OFFICE_CLASSES)

    builder._validate_canonical(office)


def test_office_alias_vocabulary_is_ordered_and_explicit() -> None:
    root = Path(builder.__file__).resolve().parents[1]
    txt_path = (
        root
        / "configs/evaluation/vocabularies/tesse_cd_office_temporal_aliases.txt"
    )
    json_path = txt_path.with_suffix(".json")

    classes = tuple(txt_path.read_text(encoding="utf-8").splitlines())
    mapping = json.loads(json_path.read_text(encoding="utf-8"))

    assert classes == OFFICE_ALIAS_CLASSES
    assert mapping == OFFICE_ALIAS_MAP
    assert set(mapping) == set(classes)
    assert set(mapping.values()) <= set(OFFICE_CLASSES)


@dataclass(frozen=True)
class Fixture:
    canonical: Path
    alias_shards: tuple[Path, Path]
    alias_ranges: tuple[str, str]
    alias_classes: tuple[Path, Path]
    alias_classes_txt: Path
    alias_configs: tuple[Path, Path]
    models: dict[str, Path]
    alias_map: Path
    output: Path

    @property
    def alias(self) -> Path:
        return self.alias_shards[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mask(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    result = np.zeros(IMAGE_SHAPE, dtype=np.bool_)
    result[y0:y1, x0:x1] = True
    return result


def _payload(
    classes: tuple[str, ...],
    rows: list[tuple[np.ndarray, int, float, float]],
) -> dict[str, object]:
    count = len(rows)
    masks = (
        np.stack([row[0] for row in rows])
        if rows
        else np.empty((0, *IMAGE_SHAPE), dtype=np.bool_)
    )
    boxes = np.empty((count, 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        boxes[index] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    seeds = np.asarray([row[3] for row in rows], dtype=np.float32)
    image_features = (
        np.stack([seeds + offset for offset in (0.0, 0.1, 0.2, 0.3)], axis=1)
        if count
        else np.empty((0, FEATURE_DIMENSION), dtype=np.float32)
    )
    text_features = (
        np.stack([seeds + offset for offset in (1.0, 1.1, 1.2, 1.3)], axis=1)
        if count
        else np.empty((0, FEATURE_DIMENSION), dtype=np.float32)
    )
    return {
        "xyxy": boxes,
        "confidence": np.asarray([row[2] for row in rows], dtype=np.float32),
        "class_id": np.asarray([row[1] for row in rows], dtype=np.int64),
        "mask": masks,
        "classes": list(classes),
        "image_crops": [None] * count,
        "image_feats": image_features,
        "text_feats": text_features,
    }


def _write_frame(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream,
    ):
        pickle.dump(payload, stream, protocol=4, fix_imports=False)


def _read_frame(path: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _cache_prefix(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for cache_index, checksum in enumerate(hashes.values()):
        digest.update(cache_index.to_bytes(8, "little", signed=False))
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _write_manifest(
    directory: Path,
    classes: tuple[str, ...],
    frame_count: int,
    *,
    algorithm_byte: str,
    feature_model_sha256: str,
) -> dict[str, object]:
    hashes = {
        f"frame{index:06d}.pkl.gz": _sha256(directory / f"frame{index:06d}.pkl.gz")
        for index in range(frame_count)
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "method": "OVIV2",
        "dataset": "TESSE-CD",
        "scene": "apartment",
        "frame_count": frame_count,
        "source_frame_ids": list(range(frame_count)),
        "source_frame_ids_hash": _json_hash(list(range(frame_count))),
        "image_shape": list(IMAGE_SHAPE),
        "class_count": len(classes),
        "classes": list(classes),
        "vocabulary_sha256": algorithm_byte * 64,
        "algorithm_hash": algorithm_byte * 64,
        "feature_model_id": f"clip-sha256:{feature_model_sha256}",
        "cache_files_sha256": hashes,
        "cache_prefix_sha256": _cache_prefix(hashes),
    }
    _write_json(directory / "frontend_manifest.json", manifest)
    return manifest


def _fixture(
    tmp_path: Path,
    *,
    canonical_rows: list[tuple[np.ndarray, int, float, float]] | None = None,
    alias_rows: list[tuple[np.ndarray, int, float, float]] | None = None,
) -> Fixture:
    canonical = tmp_path / "canonical"
    alias_shards = (tmp_path / "alias-s0", tmp_path / "alias-s1")
    alias_classes = (
        tmp_path / "alias-s0-classes.json",
        tmp_path / "alias-s1-classes.json",
    )
    alias_classes_txt = tmp_path / "alias-classes.txt"
    models = {
        "yolo_model_path": tmp_path / "yolo.pt",
        "mobile_sam_model_path": tmp_path / "mobile-sam.pt",
        "clip_pretrained_path": tmp_path / "clip.bin",
    }
    for name, path in models.items():
        path.write_bytes(f"fixture-{name}".encode())
    canonical_rows = (
        canonical_rows
        if canonical_rows is not None
        else [(_mask(0, 0, 2, 2), CANONICAL_CLASSES.index("Chair"), 0.7, 10.0)]
    )
    alias_rows = (
        alias_rows
        if alias_rows is not None
        else [
            (_mask(4, 0, 6, 2), ALIAS_CLASSES.index("book"), 0.9, 20.0),
            (_mask(4, 3, 6, 5), ALIAS_CLASSES.index("cabinet"), 0.8, 30.0),
        ]
    )
    _write_frame(
        canonical / "frame000000.pkl.gz",
        _payload(CANONICAL_CLASSES, canonical_rows),
    )
    _write_frame(canonical / "frame000001.pkl.gz", _payload(CANONICAL_CLASSES, []))
    _write_frame(
        alias_shards[0] / "frame000000.pkl.gz",
        _payload(ALIAS_CLASSES, alias_rows),
    )
    _write_frame(alias_shards[1] / "frame000001.pkl.gz", _payload(ALIAS_CLASSES, []))
    _write_manifest(
        canonical,
        CANONICAL_CLASSES,
        2,
        algorithm_byte="a",
        feature_model_sha256=_sha256(models["clip_pretrained_path"]),
    )
    alias_classes_txt.write_text("\n".join(ALIAS_CLASSES) + "\n", encoding="utf-8")
    alias_configs: list[Path] = []
    for shard_index, (start, end) in enumerate(((0, 1), (1, 2))):
        _write_json(alias_classes[shard_index], list(ALIAS_CLASSES))
        config_path = tmp_path / f"alias-s{shard_index}-config.json"
        _write_json(
            config_path,
            {
                "dataset_root": str(tmp_path / "dataset"),
                "scene_id": "apartment",
                "start": start,
                "end": end,
                "stride": 1,
                "desired_height": IMAGE_SHAPE[0],
                "desired_width": IMAGE_SHAPE[1],
                "classes_file": str(alias_classes_txt),
                "class_set": None,
                "add_bg_classes": False,
                "accumu_classes": False,
                "gsa_variant": f"alias-s{shard_index}",
                "exp_suffix": f"alias-s{shard_index}",
                "clip_model_card": "ViT-H-14",
                **{name: str(path) for name, path in models.items()},
            },
        )
        alias_configs.append(config_path)
    alias_map = tmp_path / "aliases.json"
    _write_json(alias_map, ALIAS_MAP)
    return Fixture(
        canonical=canonical,
        alias_shards=alias_shards,
        alias_ranges=("0:0", "1:1"),
        alias_classes=alias_classes,
        alias_classes_txt=alias_classes_txt,
        alias_configs=(alias_configs[0], alias_configs[1]),
        models=models,
        alias_map=alias_map,
        output=tmp_path / "temporal",
    )


def _office_fixture(tmp_path: Path) -> Fixture:
    fixture = _fixture(tmp_path, canonical_rows=[], alias_rows=[])
    for index in range(2):
        _write_frame(
            fixture.canonical / f"frame{index:06d}.pkl.gz",
            _payload(OFFICE_CLASSES, []),
        )
    canonical_manifest = _write_manifest(
        fixture.canonical,
        OFFICE_CLASSES,
        2,
        algorithm_byte="a",
        feature_model_sha256=_sha256(fixture.models["clip_pretrained_path"]),
    )
    canonical_manifest["scene"] = "office"
    _write_json(fixture.canonical / "frontend_manifest.json", canonical_manifest)

    _write_frame(
        fixture.alias_shards[0] / "frame000000.pkl.gz",
        _payload(
            OFFICE_ALIAS_CLASSES,
            [
                (
                    _mask(1, 1, 4, 4),
                    OFFICE_ALIAS_CLASSES.index("cooler"),
                    0.9,
                    20.0,
                )
            ],
        ),
    )
    _write_frame(
        fixture.alias_shards[1] / "frame000001.pkl.gz",
        _payload(OFFICE_ALIAS_CLASSES, []),
    )
    fixture.alias_classes_txt.write_text(
        "\n".join(OFFICE_ALIAS_CLASSES) + "\n", encoding="utf-8"
    )
    for classes_path, config_path in zip(
        fixture.alias_classes, fixture.alias_configs, strict=True
    ):
        _write_json(classes_path, list(OFFICE_ALIAS_CLASSES))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["scene_id"] = "office"
        _write_json(config_path, config)
    _write_json(fixture.alias_map, OFFICE_ALIAS_MAP)
    return fixture


def _run(
    fixture: Fixture,
    output: Path | None = None,
    *,
    alias_ranges: tuple[str, str] | None = None,
) -> dict[str, object]:
    argv = ["--canonical-cache-dir", str(fixture.canonical)]
    for directory, frame_range, classes, config in zip(
        fixture.alias_shards,
        alias_ranges or fixture.alias_ranges,
        fixture.alias_classes,
        fixture.alias_configs,
        strict=True,
    ):
        argv.extend(("--alias-cache-dir", str(directory)))
        argv.extend(("--alias-frame-range", frame_range))
        argv.extend(("--alias-classes", str(classes)))
        argv.extend(("--alias-config-params", str(config)))
    argv.extend(("--alias-classes-txt", str(fixture.alias_classes_txt)))
    argv.extend(("--alias-map", str(fixture.alias_map)))
    argv.extend(("--output", str(output or fixture.output)))
    args = builder.parse_args(argv)
    return builder.run(args)


def _replace_alias_frame(fixture: Fixture, payload: dict[str, object]) -> None:
    frame = fixture.alias / "frame000000.pkl.gz"
    _write_frame(frame, payload)


def test_maps_aliases_to_canonical_schema_and_preserves_clip_rows(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    manifest = _run(fixture)
    output = _read_frame(fixture.output / "frame000000.pkl.gz")

    assert output["classes"] == list(CANONICAL_CLASSES)
    assert np.asarray(output["class_id"]).tolist() == [2, 1, 5]
    assert np.asarray(output["confidence"]).tolist() == pytest.approx([0.7, 0.9, 0.8])
    assert np.asarray(output["image_feats"])[:, 0].tolist() == pytest.approx(
        [10.0, 20.0, 30.0]
    )
    assert np.asarray(output["text_feats"])[:, 0].tolist() == pytest.approx(
        [11.0, 21.0, 31.0]
    )
    assert manifest["schema_version"] == 1
    assert manifest["classes"] == list(CANONICAL_CLASSES)
    assert manifest["class_count"] == 10
    assert manifest["temporal_only"] is True


def test_builds_office_cache_end_to_end_and_maps_cooler(tmp_path: Path) -> None:
    fixture = _office_fixture(tmp_path)

    manifest = _run(fixture)
    output = _read_frame(fixture.output / "frame000000.pkl.gz")

    assert output["classes"] == list(OFFICE_CLASSES)
    assert np.asarray(output["class_id"]).tolist() == [0]
    assert np.asarray(output["image_feats"])[:, 0].tolist() == pytest.approx([20.0])
    assert manifest["scene"] == "office"
    assert manifest["classes"] == list(OFFICE_CLASSES)
    assert manifest["temporal_frontend_sources"]["alias_map"]["mapping"]["cooler"] == (
        "Small office objects"
    )


def test_same_canonical_dedup_keeps_higher_confidence_and_canonical_tie(
    tmp_path: Path,
) -> None:
    large = _mask(0, 0, 4, 4)
    contained = _mask(0, 0, 2, 2)
    separate = _mask(5, 5, 7, 7)
    fixture = _fixture(
        tmp_path,
        canonical_rows=[
            (large, CANONICAL_CLASSES.index("Chair"), 0.6, 10.0),
            (large, CANONICAL_CLASSES.index("Vase"), 0.7, 11.0),
        ],
        alias_rows=[
            (large, ALIAS_CLASSES.index("chair"), 0.8, 20.0),
            (contained, ALIAS_CLASSES.index("vase"), 0.7, 21.0),
            (large, ALIAS_CLASSES.index("book"), 0.9, 22.0),
            (separate, ALIAS_CLASSES.index("chair"), 0.5, 23.0),
        ],
    )

    _run(fixture)
    output = _read_frame(fixture.output / "frame000000.pkl.gz")

    assert np.asarray(output["class_id"]).tolist() == [3, 1, 2, 2]
    assert np.asarray(output["image_feats"])[:, 0].tolist() == pytest.approx(
        [11.0, 22.0, 20.0, 23.0]
    )


def test_alias_rows_are_stably_sorted_and_capped_at_32(tmp_path: Path) -> None:
    rows: list[tuple[np.ndarray, int, float, float]] = []
    for index in range(40):
        mask = np.zeros(IMAGE_SHAPE, dtype=np.bool_)
        mask.flat[index] = True
        confidence = 0.9 if index in (5, 7) else 0.8 - index / 1000.0
        rows.append((mask, ALIAS_CLASSES.index("book"), confidence, float(index + 1)))
    fixture = _fixture(tmp_path, canonical_rows=[], alias_rows=rows)

    manifest = _run(fixture)
    output = _read_frame(fixture.output / "frame000000.pkl.gz")

    confidences = np.asarray(output["confidence"])
    original_indices = np.asarray(output["image_feats"])[:, 0].astype(int) - 1
    assert len(confidences) == 32
    assert original_indices[:2].tolist() == [5, 7]
    assert confidences.tolist() == sorted(confidences.tolist(), reverse=True)
    assert manifest["diagnostics"]["alias_cap_dropped"] == 8


def test_output_and_manifest_are_byte_deterministic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = _run(fixture, first)
    second_manifest = _run(fixture, second)

    assert (first / "frame000000.pkl.gz").read_bytes() == (
        second / "frame000000.pkl.gz"
    ).read_bytes()
    assert (first / "frontend_manifest.json").read_bytes() == (
        second / "frontend_manifest.json"
    ).read_bytes()
    assert first_manifest == second_manifest


def test_manifest_binds_canonical_manifest_alias_shards_metadata_models_and_algorithm(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    canonical_path = fixture.canonical / "frontend_manifest.json"
    canonical_manifest = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical_manifest.update(
        {
            "input_manifest_sha256": "1" * 64,
            "input_witness": {"fixture": "canonical"},
            "provenance_sha256": {"fixture": "2" * 64},
        }
    )
    _write_json(canonical_path, canonical_manifest)

    manifest = _run(fixture)
    sources = manifest["temporal_frontend_sources"]

    assert sources["canonical"]["manifest_sha256"] == _sha256(
        canonical_path
    )
    assert (
        sources["canonical"]["cache_files_sha256"]
        == canonical_manifest["cache_files_sha256"]
    )
    assert len(sources["alias_shards"]) == 2
    for index, shard in enumerate(sources["alias_shards"]):
        frame_name = f"frame{index:06d}.pkl.gz"
        assert shard["frame_range"] == {
            "start": index,
            "end": index,
            "convention": "closed",
        }
        assert shard["cache_files_sha256"] == {
            frame_name: _sha256(fixture.alias_shards[index] / frame_name)
        }
        assert shard["classes"]["sha256"] == _sha256(fixture.alias_classes[index])
        assert shard["config_params"]["sha256"] == _sha256(fixture.alias_configs[index])
        config = json.loads(fixture.alias_configs[index].read_text(encoding="utf-8"))
        normalized = {
            key: value
            for key, value in config.items()
            if key not in {"start", "end", "gsa_variant", "exp_suffix"}
        }
        assert shard["config_params"]["generation_settings_sha256"] == _json_hash(
            normalized
        )
        for key, model_path in fixture.models.items():
            assert shard["models"][key]["sha256"] == _sha256(model_path)
    alias_hashes = {
        path.name: _sha256(path)
        for directory in fixture.alias_shards
        for path in sorted(directory.glob("frame*.pkl.gz"))
    }
    assert sources["alias_cache_prefix_sha256"] == _cache_prefix(alias_hashes)
    assert sources["alias_classes_txt"]["sha256"] == _sha256(fixture.alias_classes_txt)
    assert sources["alias_map"]["sha256"] == _sha256(fixture.alias_map)
    assert sources["alias_map"]["mapping"] == ALIAS_MAP
    assert manifest["alias_map_sha256"] == _sha256(fixture.alias_map)
    assert len(manifest["algorithm_hash"]) == 64
    assert manifest["algorithm_hash"] == manifest["merge_algorithm"]["sha256"]
    assert manifest["cache_files_sha256"]["frame000000.pkl.gz"] == _sha256(
        fixture.output / "frame000000.pkl.gz"
    )
    for field in ("input_manifest_sha256", "input_witness", "provenance_sha256"):
        assert manifest[field] == canonical_manifest[field]
    assert str(tmp_path) not in json.dumps(manifest, sort_keys=True)


def test_rejects_unmapped_alias_without_creating_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_json(fixture.alias_map, {"book": "Books"})

    with pytest.raises(ValueError, match="explicitly map every alias class"):
        _run(fixture)

    assert not fixture.output.exists()
    assert not list(tmp_path.glob(f".{fixture.output.name}.*.staging"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__("classes", ["wrong"]),
            "classes",
        ),
        (
            lambda payload: payload.__setitem__(
                "mask", np.zeros((2, 7, 8), dtype=np.bool_)
            ),
            "mask shape",
        ),
        (
            lambda payload: payload.__setitem__(
                "image_feats", np.ones((1, FEATURE_DIMENSION), dtype=np.float32)
            ),
            "feature rows",
        ),
    ],
)
def test_rejects_invalid_frame_contract_without_partial_output(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    payload = _read_frame(fixture.alias / "frame000000.pkl.gz")
    mutate(payload)
    _replace_alias_frame(fixture, payload)

    with pytest.raises(ValueError, match=message):
        _run(fixture)

    assert not fixture.output.exists()
    assert not list(tmp_path.glob(f".{fixture.output.name}.*.staging"))


@pytest.mark.parametrize("field", ["xyxy", "confidence"])
def test_rejects_values_not_representable_as_float32(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    payload = _read_frame(fixture.alias / "frame000000.pkl.gz")
    values = np.asarray(payload[field], dtype=np.float64)
    values.flat[0] = np.finfo(np.float64).max
    payload[field] = values
    _replace_alias_frame(fixture, payload)

    with pytest.raises(ValueError, match="float32"):
        _run(fixture)

    assert not fixture.output.exists()


def test_restricted_unpickler_rejects_unsafe_global(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frame = fixture.alias / "frame000000.pkl.gz"
    with frame.open("wb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", mtime=0
    ) as stream:
        pickle.dump(eval, stream, protocol=4)

    with pytest.raises(ValueError, match="invalid gzip or pickle data") as error:
        _run(fixture)

    assert isinstance(error.value.__cause__, pickle.UnpicklingError)
    assert "unsafe pickle global" in str(error.value.__cause__)
    assert not fixture.output.exists()


def test_rejects_feature_model_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.models["clip_pretrained_path"].write_bytes(b"different-clip-model")

    with pytest.raises(ValueError, match="feature_model_id"):
        _run(fixture)

    assert not fixture.output.exists()


def test_rejects_non_float32_clip_rows(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = _read_frame(fixture.alias / "frame000000.pkl.gz")
    payload["image_feats"] = np.asarray(payload["image_feats"], dtype=np.float64)
    payload["text_feats"] = np.asarray(payload["text_feats"], dtype=np.float64)
    _replace_alias_frame(fixture, payload)

    with pytest.raises(ValueError, match="float32"):
        _run(fixture)

    assert not fixture.output.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_rejects_missing_or_extra_alias_frames(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    if mutation == "missing":
        (fixture.alias / "frame000000.pkl.gz").unlink()
    else:
        _write_frame(fixture.alias / "frame000001.pkl.gz", _payload(ALIAS_CLASSES, []))

    with pytest.raises(ValueError, match="exactly cover its declared frame range"):
        _run(fixture)

    assert not fixture.output.exists()


def test_rejects_non_frame_file_inside_alias_shard(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture.alias / "unexpected.pkl.gz").write_bytes(b"not-a-frame")

    with pytest.raises(ValueError, match="exactly cover its declared frame range"):
        _run(fixture)

    assert not fixture.output.exists()


def test_rejects_overlapping_or_incomplete_shard_ranges(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(ValueError, match="exactly partition canonical cache indexes"):
        _run(fixture, alias_ranges=("0:0", "0:0"))

    assert not fixture.output.exists()


@pytest.mark.parametrize("mutation", ["classes", "source_frame_ids"])
def test_rejects_noncanonical_schema_or_frame_schedule(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture.canonical / "frontend_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "classes":
        manifest["classes"] = list(reversed(CANONICAL_CLASSES))
    else:
        manifest["source_frame_ids"] = [1, 0]
        manifest["source_frame_ids_hash"] = _json_hash([1, 0])
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="canonical .*schema|canonical .*schedule"):
        _run(fixture)

    assert not fixture.output.exists()


@pytest.mark.parametrize("mutation", ["missing", "invalid"])
def test_rejects_missing_or_invalid_canonical_vocabulary_hash(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture.canonical / "frontend_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        del manifest["vocabulary_sha256"]
    else:
        manifest["vocabulary_sha256"] = "not-a-sha256"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="vocabulary_sha256"):
        _run(fixture)

    assert not fixture.output.exists()


def test_rejects_classes_txt_json_or_config_range_disagreement(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _write_json(fixture.alias_classes[1], list(reversed(ALIAS_CLASSES)))

    with pytest.raises(ValueError, match="classes"):
        _run(fixture)

    _write_json(fixture.alias_classes[1], list(ALIAS_CLASSES))
    config = json.loads(fixture.alias_configs[0].read_text(encoding="utf-8"))
    config["end"] = 2
    _write_json(fixture.alias_configs[0], config)

    with pytest.raises(ValueError, match="frame range"):
        _run(fixture)

    assert not fixture.output.exists()


def test_existing_output_is_never_clobbered(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.output.mkdir()
    sentinel = fixture.output / "sentinel.txt"
    sentinel.write_bytes(b"owner-data")

    with pytest.raises(FileExistsError):
        _run(fixture)

    assert sentinel.read_bytes() == b"owner-data"
    assert sorted(path.name for path in fixture.output.iterdir()) == ["sentinel.txt"]
    assert not list(tmp_path.glob(f".{fixture.output.name}.*.staging"))


def test_publish_race_never_clobbers_competing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    publish = builder._publish_directory_no_replace

    def publish_after_competitor(source: Path, target: Path) -> None:
        target.mkdir()
        (target / "sentinel.txt").write_bytes(b"competitor-data")
        publish(source, target)

    monkeypatch.setattr(builder, "_publish_directory_no_replace", publish_after_competitor)

    with pytest.raises(FileExistsError):
        _run(fixture)

    sentinel = fixture.output / "sentinel.txt"
    assert sentinel.read_bytes() == b"competitor-data"
    assert sorted(path.name for path in fixture.output.iterdir()) == ["sentinel.txt"]
    assert not list(tmp_path.glob(f".{fixture.output.name}.*.staging"))
