import pytest


def test_regression_and_confirmation_require_new_unchanged_lock(tmp_path):
    from src.static_ovmap.composition_study.execution import require_access
    from src.static_ovmap.composition_study.io import write_once

    config = {
        "attempt_root": str(tmp_path),
        "binding_key": "bound",
        "spec": {
            "data": {
                "compose_cal": ["cal"],
                "regression_only": ["reg"],
                "confirmation": ["confirm1", "confirm2"],
            }
        },
        "confirmation_exposure": {"blocked_scenes": []},
    }
    assert require_access(config, "cal", "prepare-cal", "Q_GAIN") == "compose_cal"
    for scene, phase in (("reg", "regression"), ("confirm1", "confirm")):
        with pytest.raises(ValueError, match="selection"):
            require_access(config, scene, phase, "Q_GAIN")
    with pytest.raises(ValueError, match="role"):
        require_access(config, "reg", "compose-cal", "Q_GAIN")
    write_once(
        tmp_path / "selection.json",
        {
            "status": "FROZEN",
            "binding_key": "bound",
            "code": {"files": []},
            "m6_gate": {"enabled": False},
            "nomination": {"confirmation_methods": ["N0", "Q_GAIN"]},
        },
    )
    assert (
        require_access(config, "reg", "regression", "CP_M5_MIX50_NATIVE")
        == "regression_only"
    )
    assert require_access(config, "confirm1", "confirm", "Q_GAIN") == "confirmation"
    with pytest.raises(ValueError, match="method"):
        require_access(config, "confirm1", "confirm", "CP_M5_MIX50_NATIVE")
    with pytest.raises(ValueError, match="M6"):
        require_access(config, "reg", "regression", "CP_M6_MIX50_S2")
    config["confirmation_exposure"]["blocked_scenes"] = ["confirm2"]
    with pytest.raises(ValueError, match="EXPOSURE"):
        require_access(config, "confirm1", "confirm", "N0")


def test_prepare_phase_cannot_run_compositions_before_calibration(tmp_path):
    from src.static_ovmap.composition_study.execution import require_access

    config = {
        "attempt_root": str(tmp_path),
        "spec": {
            "data": {"compose_cal": ["cal"], "regression_only": [], "confirmation": []}
        },
    }
    with pytest.raises(ValueError, match="prepare"):
        require_access(config, "cal", "prepare-cal", "CP_M5_MIX50_NATIVE")
