import pytest


def test_transfer_rejects_new_methods_and_changed_temperature():
    from src.static_ovmap.replica_transfer.protocol import validate_transfer

    frozen = {"checkpoint": {"sha256": "abc"}, "final_temperatures": {
        "temperatures": {"N0": 0.01, "Q_GAIN": 0.01, "S_SIGLIP2_AREA": 0.02}}}
    config = {"scenes": ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"],
              "methods": ["N0", "Q_GAIN", "S_SIGLIP2_AREA", "CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL", "CP_M4_GAIN_S2"],
              "checkpoint": {"sha256": "abc"}, "temperatures": dict(frozen["final_temperatures"]["temperatures"]),
              "frame_ids": list(range(0, 2000, 10)), "budget": 200}
    validate_transfer(config, frozen)
    config["temperatures"]["N0"] = 0.07
    with pytest.raises(ValueError, match="temperature"):
        validate_transfer(config, frozen)
    config["temperatures"]["N0"] = 0.01
    config["methods"].append("CP_M5_MIX50_NATIVE")
    with pytest.raises(ValueError, match="method"):
        validate_transfer(config, frozen)


def test_replica_vocabulary_preserves_prediction_domain_and_ap_subset():
    from src.static_ovmap.replica_transfer.protocol import vocabulary

    names, ids, ap_ids = vocabulary("/home/ww/crove/ovimap-module-validation-upstream")
    assert len(names) == 51 and ids == tuple(range(1, 52))
    assert len(ap_ids) == 48 and set(ids) - set(ap_ids) == {1, 2, 3}
    assert names[31] == "basket"


def test_scene_binding_rejects_scannet_vocabulary_and_changed_visual_model():
    from src.static_ovmap.replica_transfer.protocol import validate_scene_binding

    model = {"identity": "native-original", "files": [{"sha256": "weights"}]}
    contract = {"checkpoint": {"sha256": "head"}, "temperatures": {"N0": .01},
                "runtime": {"cuda_device": "2"}, "gpu_lock": "/existing/lock",
                "valid_ids": list(range(1, 52)), "class_names": [str(i) for i in range(51)],
                "source_models": {"native": model}}
    local = {k: contract[k] for k in ("checkpoint", "temperatures", "runtime", "gpu_lock")}
    local["models"] = {"native": {**model, "valid_ids": contract["valid_ids"],
                                   "class_names": contract["class_names"]}}
    validate_scene_binding(local, contract)
    local["models"]["native"]["valid_ids"] = list(range(1, 201))
    with pytest.raises(ValueError, match="vocabulary"):
        validate_scene_binding(local, contract)
    local["models"]["native"]["valid_ids"] = contract["valid_ids"]
    local["models"]["native"]["identity"] = "other-model"
    with pytest.raises(ValueError, match="model"):
        validate_scene_binding(local, contract)


def test_transfer_rejects_substituted_text_binding(tmp_path, monkeypatch):
    from src.static_ovmap.replica_transfer import protocol

    contract = {"attempt_root": str(tmp_path), "frame_ids": [0], "checkpoint": {"path": "checkpoint"},
                "temperatures": {}, "runtime": {}, "gpu_lock": "lock", "valid_ids": [1],
                "class_names": ["wall"], "source_models": {"native": {"identity": "frozen", "files": []}}}
    model = {**contract["source_models"]["native"], "valid_ids": [1], "class_names": ["wall"], "text": {"path": "original"}}
    protocol.write_once(tmp_path / "text/native/model_binding.json", model)
    config = {**contract, "transfer_config": "contract", "scenes": {"room0": {"schedule": [0]}},
              "models": {"native": {**model, "text": {"path": "substitute"}}}}
    monkeypatch.setattr(protocol, "require_transfer", lambda *args: contract)
    monkeypatch.setattr(protocol.SourceIndex, "identity", lambda *args: {})
    with pytest.raises(ValueError, match="text binding"):
        protocol.require_access(config, "room0", "replica")
