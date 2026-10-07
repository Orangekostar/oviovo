"""Independent phase execution; no reuse of historical task freeze assertions."""

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
    raise NotImplementedError("development phase is not yet implemented: " + phase)
