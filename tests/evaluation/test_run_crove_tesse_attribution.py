from __future__ import annotations

import subprocess
import sys

from scripts.evaluation.run_crove_tesse_attribution import (
    _checkpoint_frame_indices,
    build_attribution_dependencies,
)
from scripts.evaluation.run_oviv2_tesse_cd_v2 import RunnerDependencies
from src.evaluation.crove_runtime_attribution import (
    CroveRuntimeAttributionProxy,
    RuntimeAttributionCapture,
)


def test_dependency_wrapper_changes_only_runtime_factory() -> None:
    runtime = object()
    base = RunnerDependencies(
        dataset_factory=lambda config: ("dataset", config),
        cache_loader_factory=lambda config, dataset: ("cache", config, dataset),
        runtime_factory=lambda config, caches: runtime,
        provenance_factory=lambda: {"commit": "a" * 40},
        environment_factory=lambda: {"python": "test"},
    )
    capture = RuntimeAttributionCapture()

    wrapped = build_attribution_dependencies(base, capture)

    assert wrapped.dataset_factory is base.dataset_factory
    assert wrapped.cache_loader_factory is base.cache_loader_factory
    assert wrapped.provenance_factory is base.provenance_factory
    assert wrapped.environment_factory is base.environment_factory
    result = wrapped.runtime_factory({}, object())
    assert isinstance(result, CroveRuntimeAttributionProxy)
    assert result._runtime is runtime
    assert result._capture is capture


def test_checkpoint_frame_indices_are_strict_and_frozen() -> None:
    assert _checkpoint_frame_indices(
        {"evaluation_checkpoint_frames": [263, 313, 363]}
    ) == frozenset({263, 313, 363})

    for value in (None, [], [313, 263], [263, 263], [True]):
        try:
            _checkpoint_frame_indices({"evaluation_checkpoint_frames": value})
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted invalid checkpoint frames: {value!r}")


def test_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/run_crove_tesse_attribution.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--attribution-output" in result.stdout
    assert "--config" in result.stdout
