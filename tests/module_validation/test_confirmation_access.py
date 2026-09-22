"""No confirmation scene or method may run without the immutable final lock."""

import json

import pytest

from src.static_ovmap.module_validation.boundary_jobs import file_identity
from src.static_ovmap.module_validation.confirmation_access import (
    confirmation_contract,
    confirmation_rows,
)


def _lock(tmp_path, candidate="S_SIGLIP2_AREA"):
    runtime, config = {"output_root": str(tmp_path / "runtime")}, {"study_root": str(tmp_path / "study")}
    source = tmp_path / "prediction.py"
    source.write_text("frozen prediction source")
    split = {"selected": [{"scene_id": f"scene{i}_00", "role": "confirm"} for i in (1, 2)]}
    frozen = tmp_path / "frozen_config.json"
    frozen.write_text(json.dumps({"study": config, "runtime": runtime, "split": split,
        "sources": [file_identity(source)], "prediction_code_commit": "a" * 40}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "FROZEN", "final_candidate": candidate,
        "confirmation_authorized": candidate != "N0", "prediction_code_commit": "a" * 40,
        "frozen_config": file_identity(frozen), "confirmation": {"status": "REQUIRED_FROZEN",
            "rows": ["N0", candidate, "S_NATIVE_AREA"]}}))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"status": "COMPLETE", "input_identity": "frozen",
        "selection_path": str(selection), "outputs": [file_identity(selection), file_identity(frozen)]}))
    return receipt, runtime, config, split, source


def test_only_exact_locked_confirmation_scenes_and_methods_are_authorized(tmp_path):
    path, runtime, config, split, source = _lock(tmp_path)
    contract = confirmation_contract(path, runtime, config)
    assert len(confirmation_rows(split, contract)) == 2
    assert confirmation_rows(split, contract, ["scene1_00"])[0]["scene_id"] == "scene1_00"
    with pytest.raises(ValueError, match="confirmation scene"):
        confirmation_rows(split, contract, ["scene3_00"])
    with pytest.raises(ValueError, match="runtime"):
        confirmation_contract(path, {"output_root": "different"}, config)
    source.write_text("changed after freeze")
    with pytest.raises(ValueError, match="source"):
        confirmation_contract(path, runtime, config)


def test_no_gain_never_authorizes_confirmation_access(tmp_path):
    path, runtime, config, _, _ = _lock(tmp_path, "N0")
    with pytest.raises(ValueError, match="retained candidate"):
        confirmation_contract(path, runtime, config)
