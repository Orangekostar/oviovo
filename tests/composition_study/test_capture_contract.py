from copy import deepcopy

import pytest


def test_confirmation_contract_allows_only_nominee_controls_and_exact_schedule():
    from src.static_ovmap.composition_study.capture_bridge import validate_contract
    from src.static_ovmap.composition_study.selection import COMPOSITIONS, CONTROLS

    rows = [
        {
            "scene_id": scene,
            "role": "confirm",
            "schedule": {
                "frame_ids": list(range(0, 400, 2)),
                "start": 0,
                "end": 400,
                "step": 2,
            },
        }
        for scene in ("scene0553_00", "scene0064_00")
    ]
    config = {
        "binding_key": "bound",
        "confirmation_rows": rows,
        "models": {},
        "checkpoint": {},
        "spec": {"data": {"confirmation": [row["scene_id"] for row in rows]}},
    }
    lock = {
        "status": "FROZEN",
        "binding_key": "bound",
        "confirmation_rows": deepcopy(rows),
        "models": {},
        "checkpoint": {},
        "spec": deepcopy(config["spec"]),
        "nomination": {
            "status": "NOMINATED",
            "nominee": COMPOSITIONS[2],
            "confirmation_methods": [*CONTROLS, COMPOSITIONS[2]],
        },
    }
    assert validate_contract(config, lock) == rows
    changed = deepcopy(lock)
    changed["nomination"]["confirmation_methods"].append(COMPOSITIONS[3])
    with pytest.raises(ValueError, match="method"):
        validate_contract(config, changed)
    changed = deepcopy(lock)
    changed["confirmation_rows"][0]["schedule"]["frame_ids"][0] = 1
    with pytest.raises(ValueError, match="schedule|rows"):
        validate_contract(config, changed)
    changed = deepcopy(lock)
    changed["nomination"]["nominee"] = COMPOSITIONS[6]
    changed["nomination"]["confirmation_methods"] = [
        *CONTROLS,
        COMPOSITIONS[6],
        COMPOSITIONS[5],
    ]
    assert validate_contract(config, changed) == rows


def test_confirmation_rejects_changed_native_runtime_even_with_same_binding_key():
    from src.static_ovmap.composition_study.capture_bridge import validate_contract
    from src.static_ovmap.composition_study.selection import COMPOSITIONS, CONTROLS

    rows = [
        {
            "scene_id": scene,
            "role": "confirm",
            "schedule": {
                "frame_ids": list(range(200)),
                "start": 0,
                "end": 200,
                "step": 1,
            },
        }
        for scene in ("scene0553_00", "scene0064_00")
    ]
    config = {
        "binding_key": "same",
        "confirmation_rows": rows,
        "models": {},
        "checkpoint": {},
        "runtime": {"native_model": "changed"},
        "source": {},
        "gpu_lock": "original",
        "spec": {"data": {"confirmation": [row["scene_id"] for row in rows]}},
    }
    lock = {
        "status": "FROZEN",
        **deepcopy(config),
        "runtime": {"native_model": "frozen"},
        "nomination": {
            "status": "NOMINATED",
            "nominee": COMPOSITIONS[0],
            "confirmation_methods": [*CONTROLS, COMPOSITIONS[0]],
        },
    }
    with pytest.raises(ValueError, match="runtime"):
        validate_contract(config, lock)
