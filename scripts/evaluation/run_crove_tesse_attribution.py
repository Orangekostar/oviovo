#!/usr/bin/env python3
"""Run the Apartment CROVE v2 path with instrumentation-only attribution."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import run_oviv2_tesse_cd_v2 as _runner
from src.evaluation.crove_runtime_attribution import (
    CroveRuntimeAttributionProxy,
    RuntimeAttributionCapture,
    publish_runtime_attribution,
)
from src.evaluation.json_contracts import loads_strict


def _checkpoint_frame_indices(
    payload: Mapping[str, object],
) -> frozenset[int]:
    values = payload.get("evaluation_checkpoint_frames")
    if not isinstance(values, list) or not values:
        raise ValueError("evaluation checkpoint frames must be a nonempty list")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("evaluation checkpoint frames must be nonnegative integers")
    if values != sorted(set(values)):
        raise ValueError("evaluation checkpoint frames must increase strictly")
    return frozenset(values)


def build_attribution_dependencies(
    base: _runner.RunnerDependencies,
    capture: RuntimeAttributionCapture,
) -> _runner.RunnerDependencies:
    """Preserve production dependencies and instrument only the runtime leaf."""

    if not isinstance(base, _runner.RunnerDependencies):
        raise TypeError("base must be RunnerDependencies")
    if not isinstance(capture, RuntimeAttributionCapture):
        raise TypeError("capture must be RuntimeAttributionCapture")

    def runtime_factory(config: dict[str, Any], caches: Any) -> Any:
        return CroveRuntimeAttributionProxy(
            base.runtime_factory(config, caches), capture
        )

    return _runner.RunnerDependencies(
        dataset_factory=base.dataset_factory,
        cache_loader_factory=base.cache_loader_factory,
        runtime_factory=runtime_factory,
        provenance_factory=base.provenance_factory,
        environment_factory=base.environment_factory,
    )


def run_attribution(
    config: str | Path,
    output: str | Path,
    attribution_output: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Execute the existing causal runner and publish a separate sidecar."""

    config_path = Path(config).resolve(strict=True)
    payload = loads_strict(
        config_path.read_text(encoding="utf-8"), label="attribution config"
    )
    if not isinstance(payload, dict) or payload.get("scene") != "apartment":
        raise ValueError("runtime attribution is restricted to Apartment")
    temporal = payload.get("temporal_readout")
    if not isinstance(temporal, dict) or temporal.get("execution_profile") != "a4":
        raise ValueError("runtime attribution requires the Apartment A4 profile")
    output_path = Path(output).resolve()
    attribution_path = Path(attribution_output).resolve()
    if attribution_path.is_relative_to(output_path) or output_path.is_relative_to(
        attribution_path
    ):
        raise ValueError("attribution output must be separate from the formal run")

    capture = RuntimeAttributionCapture(
        capture_frame_indices=_checkpoint_frame_indices(payload)
    )
    dependencies = build_attribution_dependencies(
        _runner._production_dependencies(), capture
    )
    manifest = _runner.run(
        config_path,
        output_path,
        causal_table_metrics_only=True,
        dependencies=dependencies,
    )
    published = publish_runtime_attribution(
        attribution_path,
        records=capture.records,
        run_manifest=output_path / "run_manifest.json",
        instrumentation_sources={
            "launcher": Path(__file__).resolve(),
            "runtime_attribution": ROOT / "src/evaluation/crove_runtime_attribution.py",
        },
    )
    return manifest, published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attribution-output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest, attribution = run_attribution(
        args.config, args.output, args.attribution_output
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "attribution_output": str(attribution),
                "processed_frame_count": manifest["processed_frame_count"],
                "scene": manifest["scene"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
