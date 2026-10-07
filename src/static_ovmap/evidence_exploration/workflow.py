"""Independent phase execution; no reuse of historical task freeze assertions."""


def completed_acquisition(binding, output_root):
    from collections import Counter
    from pathlib import Path
    from static_ovmap.cvpr_compact.projected_views import _verified_identity
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
    from .evaluation import require_freeze
    root = Path(output_root)
    if not (root/"encoded/summary.json").exists() or not (root/"implementation_freeze.json").exists():
        return None
    require_freeze(root)
    summary, index, counts = read(root/"encoded/summary.json"), ConsumptionIndex(), Counter()
    _verified_identity(summary)
    prepared = {scene: read(root/"prepared"/scene/"receipt.json") for scene in binding["scenes"]}
    for row in prepared.values():
        _verified_identity(row)
        if row["binding_identity"] != binding["identity"]:
            raise ValueError("completed geometry belongs to a different task binding")
    expected = {(scene, int(fid)) for scene, row in prepared.items() for fid, frame in row["frames"].items() if frame["triggered_owners"]}
    if summary["status"] != "ENCODED" or {(x["scene"], int(x["frame_id"])) for x in summary["frames"]} != expected:
        raise ValueError("completed scientific acquisition summary has incomplete coverage")
    for item in summary["frames"]:
        row = read(root/"encoded"/item["scene"]/str(item["frame_id"])/"receipt.json")
        _verified_identity(row)
        if row["identity"] != item["identity"] or row["prepared_identity"] != prepared[item["scene"]]["identity"]:
            raise ValueError("completed acquired feature or prepared-input identity changed")
        index.identity(row["vectors"]["path"], row["vectors"])
        counts.update(row["counts"])
    if dict(counts) != summary["counts"]:
        raise ValueError("completed acquired inference counters changed")
    print("ENCODING_REUSE", len(summary["frames"]), "NO_NEW_INFERENCE_OR_MODEL_LOAD", flush=True)
    return summary


def completed_timing(binding, output_root):
    """Resume completed calls without reloading models or rewriting load costs."""
    from pathlib import Path
    from static_ovmap.cvpr_compact.area_fallback_experiment import seal
    from static_ovmap.cvpr_compact.projected_views import _verified_identity
    from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
    from .evaluation import require_freeze
    root = Path(output_root)
    if not (root/"timing/summary.json").exists():
        return None
    require_freeze(root)
    selection = read(root/"selection.json")
    _verified_identity(selection)
    expected = seal({"selection_identity": selection["identity"],
        "source": ConsumptionIndex().identity(Path(__file__).with_name("timing.py")),
        "method_order": selection["timing_methods"], "repeats": 2,
        "scene_order": binding["cohorts"]["replica8"]})
    stored_binding, summary = read(root/"timing/binding.json"), read(root/"timing/summary.json")
    _verified_identity(stored_binding)
    _verified_identity(summary)
    if stored_binding != expected or summary["binding_identity"] != expected["identity"]:
        raise ValueError("completed timing source or metric-selection identity changed")
    if summary["status"] != "PAIRED_COLD_TIMING_COMPLETE" or len(summary["calls"]) != 16*len(selection["timing_methods"]):
        raise ValueError("existing completed timing summary has incomplete coverage")
    for row in summary["calls"]:
        path = root/"timing/calls"/(str(row["repeat"])+"_"+row["scene"]+"_"+row["method"])/"receipt.json"
        current = read(path)
        _verified_identity(current)
        locked = read(root/"predictions"/row["scene"]/"receipt.json")["outcomes"][row["method"]]
        if current != row or current["prediction_content_identity"] != locked["content_identity"]:
            raise ValueError("completed cold receipt or prediction changed")
    print("TIMING_REUSE", len(summary["calls"]), "NO_NEW_INFERENCE_OR_MODEL_LOAD", flush=True)
    return summary


def run_phase(binding, output_root, phase, *, resume=False):
    if phase == "diagnose":
        from .analysis import diagnose
        from .diagnostic_details import supplement_baselines
        result = diagnose(binding, output_root)
        supplement_baselines(binding, output_root)
        return result
    if phase == "prepare":
        from .preparation import prepare
        return prepare(binding, output_root)
    if phase == "encode":
        previous = completed_acquisition(binding, output_root)
        if previous is not None:
            return previous
        import os
        import subprocess
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": binding["gpu"], "OMP_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        subprocess.run([binding["FC_python"], "-m", "static_ovmap.evidence_exploration.acquisition_worker",
                        "--output-root", str(output_root)], env=env, check=True)
        return
    if phase == "predict":
        from .outputs import predict
        return predict(binding, output_root)
    if phase == "evaluate":
        from .freeze import freeze
        from .evaluation import evaluate
        freeze(binding, output_root)
        return evaluate(binding, output_root)
    if phase == "select":
        from .selection import select
        return select(binding, output_root)
    if phase == "time":
        previous = completed_timing(binding, output_root)
        if previous is not None:
            return previous
        import os
        import subprocess
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": binding["gpu"], "OMP_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        subprocess.run([binding["FC_python"], "-m", "static_ovmap.evidence_exploration.timing_worker",
                        "--output-root", str(output_root)], env=env, check=True)
        return
    if phase == "tables":
        from .publication import build_tables
        return build_tables(binding, output_root)
    if phase == "publish":
        from .publication import publish
        return publish(binding, output_root)
    raise NotImplementedError("development phase is not yet implemented: " + phase)
