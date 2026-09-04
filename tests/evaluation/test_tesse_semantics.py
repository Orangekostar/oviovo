from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from src.evaluation.baselines.tesse_semantics import (
    load_tesse_semantic_crosswalk,
    normalize_semantic_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIASES = REPO_ROOT / "configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml"
LABEL_ROOT = Path(
    "/home/ww/oviovo_baseline_builds/khronos-jazzy-ws/src/hydra/config/label_spaces"
)

FROZEN_LABELS = {
    "Appliance",
    "Bed",
    "Chair",
    "Couch",
    "Deformable",
    "Door",
    "Light",
    "Plant",
    "Shelf",
    "Storage",
    "Table",
    "Thing",
    "Wall_Decoration",
}


def _write_label_space(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "total_semantic_labels": 4,
                "object_labels": [1, 2],
                "label_names": [
                    {"label": 0, "name": "Unknown"},
                    {"label": 1, "name": "Chair"},
                    {"label": 2, "name": "Table"},
                    {"label": 3, "name": "Floor"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_aliases(path: Path, label_space: Path, aliases: dict[str, list[str]]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "protocol": "tesse_cd_common_v2",
                "unknown_name": "Unknown",
                "scenes": {
                    "apartment": {
                        "label_space": {
                            "filename": label_space.name,
                            "sha256": hashlib.sha256(label_space.read_bytes()).hexdigest(),
                        },
                        "aliases": aliases,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_crosswalk_maps_names_and_unmatched_values_to_unknown(tmp_path: Path) -> None:
    label_space = tmp_path / "labels.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_label_space(label_space)
    _write_aliases(
        aliases,
        label_space,
        {"Chair": ["chair", "chairs"], "Table": ["table", "dining table"]},
    )

    crosswalk = load_tesse_semantic_crosswalk(aliases, "apartment", label_space)

    assert crosswalk.lookup(" Dining_Table ").semantic_id == 2
    assert crosswalk.lookup("CHAIRS").native_name == "Chair"
    assert crosswalk.lookup("sofa").semantic_id == 0
    assert crosswalk.lookup("sofa").matched is False
    assert crosswalk.valid_semantic_ids == frozenset({1, 2})


def test_crosswalk_recognizes_canonical_nonobject_label_without_object_alias(
    tmp_path: Path,
) -> None:
    label_space = tmp_path / "labels.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_label_space(label_space)
    _write_aliases(aliases, label_space, {"Chair": ["chair"]})

    crosswalk = load_tesse_semantic_crosswalk(aliases, "apartment", label_space)

    floor = crosswalk.lookup(" floor ")
    assert floor.semantic_id == 3
    assert floor.native_name == "Floor"
    assert floor.matched is True
    assert floor.semantic_id not in crosswalk.valid_semantic_ids


def test_crosswalk_rejects_direct_numeric_id_comparison(tmp_path: Path) -> None:
    label_space = tmp_path / "labels.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_label_space(label_space)
    _write_aliases(aliases, label_space, {"Chair": ["chair"]})
    crosswalk = load_tesse_semantic_crosswalk(aliases, "apartment", label_space)

    with pytest.raises(TypeError, match="numeric semantic IDs"):
        crosswalk.lookup(1)


def test_crosswalk_rejects_duplicate_normalized_aliases(tmp_path: Path) -> None:
    label_space = tmp_path / "labels.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_label_space(label_space)
    _write_aliases(
        aliases,
        label_space,
        {"Chair": ["wall-decoration"], "Table": ["wall decoration"]},
    )

    with pytest.raises(ValueError, match="duplicate normalized alias"):
        load_tesse_semantic_crosswalk(aliases, "apartment", label_space)


def test_crosswalk_rejects_unknown_target_and_label_space_hash_drift(
    tmp_path: Path,
) -> None:
    label_space = tmp_path / "labels.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_label_space(label_space)
    _write_aliases(aliases, label_space, {"Sofa": ["sofa"]})

    with pytest.raises(ValueError, match="alias target is not an object label"):
        load_tesse_semantic_crosswalk(aliases, "apartment", label_space)

    _write_aliases(aliases, label_space, {"Chair": ["chair"]})
    label_space.write_text(label_space.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_tesse_semantic_crosswalk(aliases, "apartment", label_space)


@pytest.mark.skipif(not LABEL_ROOT.is_dir(), reason="pinned Khronos label spaces unavailable")
@pytest.mark.parametrize("scene", ["apartment", "office"])
def test_checked_crosswalk_covers_frozen_label_inventory(scene: str) -> None:
    label_space = LABEL_ROOT / f"tesse_cd_{scene}_label_space.yaml"
    crosswalk = load_tesse_semantic_crosswalk(ALIASES, scene, label_space)

    mapped = {label: crosswalk.lookup(label) for label in FROZEN_LABELS}
    assert all(result.semantic_id >= 0 for result in mapped.values())
    assert mapped["Chair"].matched is True
    if scene == "apartment":
        assert mapped["Chair"].native_name == "Chair"
        assert mapped["Couch"].native_name == "Couch"
        assert mapped["Table"].native_name == "Table"
    else:
        assert mapped["Chair"].native_name == "Chairs"
        assert mapped["Couch"].matched is False
        assert mapped["Table"].matched is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Wall_Decoration", "walldecoration"), (" dining-table ", "diningtable")],
)
def test_normalize_semantic_name(raw: str, expected: str) -> None:
    assert normalize_semantic_name(raw) == expected
