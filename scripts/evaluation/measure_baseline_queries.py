#!/usr/bin/env python3
"""Measure native text-to-entity query latency from a neutral baseline artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _query_model_name(baseline: str) -> str:
    return {
        "conceptgraphs": "ViT-H-14",
        "dualmap": "MobileCLIP-S2",
        "ovimap": "SigLIP-L/16-384",
    }[baseline]


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
    return np.vstack(eligible)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("conceptgraphs", "dualmap", "ovimap"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from src.evaluation.exporters.oviovo import read_map_snapshot

    io_started = time.perf_counter()
    snapshot = read_map_snapshot(args.snapshot, args.entities)
    embeddings = _entity_embeddings(snapshot.entities, args.baseline)
    queries = [
        str(value)
        for value in json.loads(args.queries.read_text(encoding="utf-8"))["vocabulary"]["classes"]
    ]
    evaluation_io_s = time.perf_counter() - io_started

    model_name = _query_model_name(args.baseline)
    initialized = time.perf_counter()
    if args.baseline == "ovimap":
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            str(args.clip_weight),
            local_files_only=True,
        ).to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.clip_weight),
            local_files_only=True,
        )
    else:
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=str(args.clip_weight),
        )
        model = model.to(args.device).eval()
        if args.baseline == "dualmap":
            from mobileclip.modules.common.mobileone import reparameterize_model

            model = reparameterize_model(model)
        tokenizer = open_clip.get_tokenizer(model_name)
    entity_features = torch.as_tensor(embeddings, dtype=torch.float32, device=args.device)
    entity_features /= entity_features.norm(dim=-1, keepdim=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    initialization_s = time.perf_counter() - initialized

    def run_query(label: str) -> None:
        if args.baseline == "ovimap":
            tokens = tokenizer(
                [label],
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to(args.device)
        else:
            text = label if args.baseline == "dualmap" else f"an image of {label}"
            tokens = tokenizer([text]).to(args.device)
        with torch.no_grad():
            feature = (
                model.get_text_features(**tokens)
                if args.baseline == "ovimap"
                else model.encode_text(tokens)
            ).float()
            feature /= feature.norm(dim=-1, keepdim=True)
            torch.argmax(feature @ entity_features.T, dim=1)

    for index in range(args.warmup):
        run_query(queries[index % len(queries)])
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    latencies_ms = []
    for query in queries:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        run_query(query)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    payload = {
        "baseline": args.baseline,
        "scene_id": snapshot.scene_id,
        "query_count": len(queries),
        "warmup_count": args.warmup,
        "entity_count": len(embeddings),
        "model": model_name,
        "device": args.device,
        "initialization_s": initialization_s,
        "evaluation_io_s": evaluation_io_s,
        "latencies_ms": latencies_ms,
        "query_p50_ms": _percentile(latencies_ms, 50.0),
        "query_p95_ms": _percentile(latencies_ms, 95.0),
        "protocol": (
            "frozen Replica-41 category queries; full tokenization, text encoding, and entity ranking; "
            "OVI-MAP excludes instances observed in fewer than two frames"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
