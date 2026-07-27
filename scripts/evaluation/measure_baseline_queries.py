#!/usr/bin/env python3
"""Measure native text-to-entity query latency from a neutral baseline artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


_OVIV2_WARMUP_COUNT = 10
_OVIV2_MEASURED_REPEATS = 5
_DEFAULT_OVIV2_PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "configs/evaluation/manifests/oviv2_tesse_t4_v1.json"
)
_OVIV2_OPERATION_ORDER = [
    "full_tokenization",
    "text_encoding",
    "l2_normalization",
    "current_map_entity_ranking",
]


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _query_model_name(baseline: str) -> str:
    return {
        "conceptgraphs": "ViT-H-14",
        "dualmap": "MobileCLIP-S2",
        "ovimap": "SigLIP-L/16-384",
        "oviv2": "ViT-H-14",
    }[baseline]


def _query_protocol(baseline: str) -> str:
    protocol = "frozen Replica-41 category queries; full tokenization, text encoding, and entity ranking"
    if baseline == "ovimap":
        protocol += "; OVI-MAP excludes instances observed in fewer than two frames"
    if baseline == "oviv2":
        protocol += "; selected current-map entities; no precomputed query embeddings"
    return protocol


def _entity_embeddings(entities: list[object], baseline: str) -> np.ndarray:
    eligible = [
        entity.semantic_embedding
        for entity in entities
        if entity.semantic_embedding is not None
        and (
            baseline != "ovimap"
            or int(entity.metadata.get("observation_count", 0)) >= 2
        )
    ]
    if not eligible:
        raise ValueError(f"{baseline} snapshot contains no query-eligible entity embeddings")
    embeddings = np.vstack(eligible).astype(np.float32, copy=False)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError(f"{baseline} entity embeddings must be a finite matrix")
    if np.any(np.linalg.norm(embeddings.astype(np.float64), axis=1) == 0.0):
        raise ValueError(f"{baseline} entity embeddings must have non-zero norm")
    return embeddings


def _source_record(
    path: Path, *, allow_directory: bool = False
) -> tuple[bytes, dict[str, object]]:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        content = resolved.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        byte_count = len(content)
    elif allow_directory and resolved.is_dir():
        tree_digest = hashlib.sha256()
        byte_count = 0
        for member in sorted(resolved.rglob("*")):
            if member.is_symlink() or (not member.is_dir() and not member.is_file()):
                raise ValueError(f"model checkpoint directory has invalid member: {member}")
            if member.is_dir():
                continue
            relative = member.relative_to(resolved).as_posix().encode("utf-8")
            member_content = member.read_bytes()
            tree_digest.update(len(relative).to_bytes(8, "big"))
            tree_digest.update(relative)
            tree_digest.update(len(member_content).to_bytes(8, "big"))
            tree_digest.update(member_content)
            byte_count += len(member_content)
        content = b""
        digest = tree_digest.hexdigest()
    else:
        raise ValueError(f"query measurement source is not a regular file: {path}")
    return content, {
        "path": str(resolved),
        "sha256": digest,
        "byte_count": byte_count,
    }


def _revalidate_source(
    role: str, record: dict[str, object], *, allow_directory: bool = False
) -> None:
    path = Path(str(record["path"]))
    try:
        _, current = _source_record(path, allow_directory=allow_directory)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{role} source changed during query measurement") from exc
    if (
        current["byte_count"] != record["byte_count"]
        or current["sha256"] != record["sha256"]
    ):
        raise ValueError(f"{role} source changed during query measurement")


def _load_queries(content: bytes) -> list[str]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=lambda pairs: _strict_object(pairs, "query vocabulary"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite query vocabulary value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("query vocabulary is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("query vocabulary must be an object")
    nested = payload.get("vocabulary")
    nested_classes = nested.get("classes") if isinstance(nested, dict) else None
    top_level_classes = payload.get("classes")
    if (nested_classes is None) == (top_level_classes is None):
        raise ValueError(
            "query vocabulary must contain exactly one classes or vocabulary.classes field"
        )
    classes = top_level_classes if top_level_classes is not None else nested_classes
    if not isinstance(classes, list) or not classes:
        raise ValueError("query vocabulary classes must be a non-empty list")
    queries: list[str] = []
    for value in classes:
        if type(value) is not str or not value.strip() or value != value.strip():
            raise ValueError("query vocabulary classes must be non-empty canonical strings")
        queries.append(value)
    if len(set(queries)) != len(queries):
        raise ValueError("query vocabulary classes must be unique")
    return queries


def _strict_object(pairs: list[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate {label} key: {key}")
        result[key] = value
    return result


def _validate_oviv2_protocol(
    content: bytes,
    *,
    warmup_count: int,
    measured_repeats: int,
    query_sha256: object,
    checkpoint_sha256: object,
) -> None:
    try:
        protocol = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=lambda pairs: _strict_object(pairs, "OVIV2 protocol"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite OVIV2 protocol value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OVIV2 query protocol is invalid JSON") from exc
    query = protocol.get("query") if isinstance(protocol, dict) else None
    expected_fields = {
        "vocabulary_path",
        "vocabulary_sha256",
        "text_model_id",
        "checkpoint_sha256",
        "operation_order",
        "warmup_count",
        "measured_repeats",
        "cuda_synchronize_each_query",
        "precomputed_query_embeddings",
    }
    if not isinstance(query, dict) or set(query) != expected_fields:
        raise ValueError("OVIV2 query protocol schema is invalid")
    if query["text_model_id"] != "ViT-H-14":
        raise ValueError("OVIV2 query protocol text model is not ViT-H-14")
    if query["operation_order"] != _OVIV2_OPERATION_ORDER:
        raise ValueError("OVIV2 query protocol operation order is invalid")
    if query["cuda_synchronize_each_query"] is not True:
        raise ValueError("OVIV2 query protocol must synchronize every CUDA query")
    if query["precomputed_query_embeddings"] is not False:
        raise ValueError("OVIV2 query protocol forbids precomputed query embeddings")
    if query["warmup_count"] != warmup_count:
        raise ValueError("oviv2 protocol requires warmup_count=10")
    if query["measured_repeats"] != measured_repeats:
        raise ValueError("oviv2 protocol requires measured repeats=5")
    if query["warmup_count"] != _OVIV2_WARMUP_COUNT:
        raise ValueError("OVIV2 frozen protocol warmup count is invalid")
    if query["measured_repeats"] != _OVIV2_MEASURED_REPEATS:
        raise ValueError("OVIV2 frozen protocol repeat count is invalid")
    if query["vocabulary_sha256"] != query_sha256:
        raise ValueError("OVIV2 query vocabulary SHA-256 does not match frozen protocol")
    if query["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("OVIV2 checkpoint SHA-256 does not match frozen protocol")


def _query_text(baseline: str, label: str) -> str:
    return label if baseline in {"dualmap", "ovimap"} else f"an image of {label}"


class _TorchQueryBackend:
    def __init__(
        self, baseline: str, model_name: str, checkpoint: Path, device: str
    ) -> None:
        import torch

        self._baseline = baseline
        self._device = device
        self._torch = torch
        if baseline == "ovimap":
            from transformers import AutoModel, AutoTokenizer

            self._model = AutoModel.from_pretrained(
                str(checkpoint), local_files_only=True
            ).to(device).eval()
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint), local_files_only=True
            )
        else:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=str(checkpoint)
            )
            model = model.to(device).eval()
            if baseline == "dualmap":
                from mobileclip.modules.common.mobileone import reparameterize_model

                model = reparameterize_model(model)
            self._model = model
            self._tokenizer = open_clip.get_tokenizer(model_name)

    def prepare_entities(self, embeddings: np.ndarray) -> Any:
        features = self._torch.as_tensor(
            embeddings, dtype=self._torch.float32, device=self._device
        )
        return features / features.norm(dim=-1, keepdim=True)

    def synchronize(self) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.synchronize(self._device)

    def tokenize(self, text: str) -> Any:
        if self._baseline == "ovimap":
            return self._tokenizer(
                [text], padding="max_length", max_length=64, return_tensors="pt"
            ).to(self._device)
        return self._tokenizer([text]).to(self._device)

    def encode_text(self, tokens: Any) -> Any:
        with self._torch.no_grad():
            feature = (
                self._model.get_text_features(**tokens)
                if self._baseline == "ovimap"
                else self._model.encode_text(tokens)
            ).float()
        norm = feature.norm(dim=-1, keepdim=True)
        if not bool(self._torch.isfinite(norm).all()) or bool((norm == 0).any()):
            raise ValueError("text encoder produced an invalid feature")
        return feature / norm

    def rank_entities(self, feature: Any, entity_features: Any) -> int:
        with self._torch.no_grad():
            top = self._torch.argmax(feature @ entity_features.T, dim=1)
        return int(top.item())


def _default_snapshot_loader(snapshot: Path, entities: Path) -> Any:
    from src.evaluation.exporters.oviovo import read_map_snapshot

    return read_map_snapshot(snapshot, entities)


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: Callable[[str, str, Path, str], Any] = _TorchQueryBackend,
    snapshot_loader: Callable[[Path, Path], Any] = _default_snapshot_loader,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=("conceptgraphs", "dualmap", "ovimap", "oviv2"),
        required=True,
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=_positive_integer, default=5)
    parser.add_argument("--repeats", type=_positive_integer, default=1)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.baseline == "oviv2":
        protocol_path = args.protocol or _DEFAULT_OVIV2_PROTOCOL
    elif args.protocol is not None:
        raise ValueError("--protocol is only valid for the oviv2 baseline")
    else:
        protocol_path = None

    io_started = clock()
    source_contents: dict[str, bytes] = {}
    sources: dict[str, dict[str, object]] = {}
    for role, path in (
        ("snapshot", args.snapshot),
        ("entities", args.entities),
        ("queries", args.queries),
        ("checkpoint", args.clip_weight),
    ):
        source_contents[role], sources[role] = _source_record(
            path,
            allow_directory=role == "checkpoint" and args.baseline == "ovimap",
        )
    protocol_record = None
    if protocol_path is not None:
        protocol_content, protocol_record = _source_record(protocol_path)
        _validate_oviv2_protocol(
            protocol_content,
            warmup_count=args.warmup,
            measured_repeats=args.repeats,
            query_sha256=sources["queries"]["sha256"],
            checkpoint_sha256=sources["checkpoint"]["sha256"],
        )
    snapshot = snapshot_loader(
        Path(str(sources["snapshot"]["path"])),
        Path(str(sources["entities"]["path"])),
    )
    for role in ("snapshot", "entities"):
        _revalidate_source(role, sources[role])
    if args.baseline == "oviv2" and snapshot.scope != "current":
        raise ValueError("oviv2 query measurement requires a selected current-map snapshot")
    embeddings = _entity_embeddings(snapshot.entities, args.baseline)
    queries = _load_queries(source_contents["queries"])
    evaluation_io_s = clock() - io_started

    model_name = _query_model_name(args.baseline)
    initialized = clock()
    backend = backend_factory(
        args.baseline,
        model_name,
        Path(str(sources["checkpoint"]["path"])),
        args.device,
    )
    entity_features = backend.prepare_entities(embeddings)
    backend.synchronize()
    initialization_s = clock() - initialized
    _revalidate_source(
        "checkpoint", sources["checkpoint"], allow_directory=args.baseline == "ovimap"
    )

    def run_query(label: str) -> int:
        tokens = backend.tokenize(_query_text(args.baseline, label))
        feature = backend.encode_text(tokens)
        top_entity_index = backend.rank_entities(feature, entity_features)
        if type(top_entity_index) is not int or not 0 <= top_entity_index < len(embeddings):
            raise ValueError("entity ranking returned an invalid top entity index")
        return top_entity_index

    for index in range(args.warmup):
        backend.synchronize()
        run_query(queries[index % len(queries)])
        backend.synchronize()

    samples: list[dict[str, int | float]] = []
    latencies_ms: list[float] = []
    for repeat_index in range(args.repeats):
        for query_index, query in enumerate(queries):
            backend.synchronize()
            started = clock()
            top_entity_index = run_query(query)
            backend.synchronize()
            latency_ms = (clock() - started) * 1000.0
            if not np.isfinite(latency_ms) or latency_ms < 0.0:
                raise ValueError("query latency must be finite and non-negative")
            latencies_ms.append(latency_ms)
            samples.append({
                "repeat_index": repeat_index,
                "query_index": query_index,
                "latency_ms": latency_ms,
                "top_entity_index": top_entity_index,
            })

    for role in ("snapshot", "entities", "queries", "checkpoint"):
        _revalidate_source(
            role,
            sources[role],
            allow_directory=role == "checkpoint" and args.baseline == "ovimap",
        )
    if protocol_record is not None:
        _revalidate_source("protocol", protocol_record)
    query_mean_ms = float(np.mean(np.asarray(latencies_ms, dtype=np.float64)))
    query_p50_ms = _percentile(latencies_ms, 50.0)
    query_p95_ms = _percentile(latencies_ms, 95.0)
    payload = {
        "schema_version": 1,
        "manifest_id": f"{args.baseline}-query-measurement-v1",
        "baseline": args.baseline,
        "scene_id": snapshot.scene_id,
        "query_count": len(queries),
        "warmup_count": args.warmup,
        "measured_repeats": args.repeats,
        "repeat_count": args.repeats,
        "entity_count": len(embeddings),
        "model": model_name,
        "text_model": {
            "model_id": model_name,
            "checkpoint_sha256": sources["checkpoint"]["sha256"],
        },
        "query_vocabulary_sha256": sources["queries"]["sha256"],
        **({"query_protocol": protocol_record} if protocol_record is not None else {}),
        "device": args.device,
        "initialization_s": initialization_s,
        "evaluation_io_s": evaluation_io_s,
        "latencies_ms": latencies_ms,
        "samples": samples,
        "query_mean_ms": query_mean_ms,
        "query_p50_ms": query_p50_ms,
        "query_p95_ms": query_p95_ms,
        "summary": {
            "query_mean_ms": query_mean_ms,
            "query_p50_ms": query_p50_ms,
            "query_p95_ms": query_p95_ms,
        },
        "sources": sources,
        "protocol": _query_protocol(args.baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
