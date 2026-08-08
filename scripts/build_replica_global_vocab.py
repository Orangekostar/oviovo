#!/usr/bin/env python3
"""Build a stable global Replica vocabulary for selected scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLICA_ORIGINAL_ROOT = Path("/home/ww/vv/dataset/Replica-Dataset/Replica_original")
DEFAULT_OUTPUT_PATH = REPO_ROOT / "configs" / "replica_global_vocab.yaml"
DEFAULT_SUMMARY_OUTPUT_PATH = REPO_ROOT / "outputs" / "tmp_validation" / "replica_global_vocab_summary.json"

SELECTED_SCENES: dict[str, str] = {
    "room0": "room_0",
    "room1": "room_1",
    "room2": "room_2",
    "office0": "office_0",
    "office1": "office_1",
    "office2": "office_2",
    "office3": "office_3",
    "office4": "office_4",
}

STRUCTURAL_LABEL_CANDIDATES = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "blinds",
    "rug",
    "pillar",
    "vent",
}

BANNED_BROAD_PROMPTS = {
    "object",
    "thing",
    "furniture",
    "decor",
    "background",
}

CONTAMINANT_LABELS = {
    "undefined",
    "other-leaf",
    "anonymize_picture",
    "anonymize_text",
    "non-plane",
}

CONSERVATIVE_PROMPT_ALIASES: dict[str, list[str]] = {
    "sofa": ["sofa", "couch"],
    "rug": ["rug", "carpet"],
    "picture": ["picture", "painting", "wall art"],
    "wall-plug": ["wall plug", "outlet", "electrical outlet"],
    "indoor-plant": ["indoor plant", "plant"],
    "cabinet": ["cabinet", "cupboard"],
    "blinds": ["blinds", "window blinds"],
}

CLASS_CONTAINER_KEYS = (
    "semantic_classes",
    "categories",
    "objects",
    "instances",
    "labels",
)
NAME_KEYS = (
    "class_name",
    "category_name",
    "semantic_label",
    "raw_category",
    "label",
    "name",
)
NESTED_SEMANTIC_KEYS = ("category", "class", "semantic_class", "semantic_category")


def _canonicalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    return label or None


def _extract_label_from_mapping(payload: dict[str, Any]) -> str | None:
    for key in NAME_KEYS[:-1]:
        label = _canonicalize_label(payload.get(key))
        if label is not None:
            return label
    for key in NESTED_SEMANTIC_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            label = _extract_label_from_mapping(value)
            if label is not None:
                return label
        else:
            label = _canonicalize_label(value)
            if label is not None:
                return label
    label = _canonicalize_label(payload.get("name"))
    if label is not None:
        return label
    return None


def _labels_from_node(node: Any) -> set[str]:
    labels: set[str] = set()
    if isinstance(node, str):
        label = _canonicalize_label(node)
        if label is not None:
            labels.add(label)
        return labels
    if isinstance(node, list):
        for item in node:
            labels.update(_labels_from_node(item))
        return labels
    if isinstance(node, dict):
        label = _extract_label_from_mapping(node)
        if label is not None:
            labels.add(label)
        else:
            for value in node.values():
                labels.update(_labels_from_node(value))
        return labels
    return labels


def _extract_class_names(info_semantic: dict[str, Any]) -> list[str]:
    if "classes" in info_semantic:
        labels = _labels_from_node(info_semantic["classes"])
        return sorted(label for label in labels if label not in CONTAMINANT_LABELS)

    labels: set[str] = set()
    for key in CLASS_CONTAINER_KEYS:
        if key in info_semantic:
            labels.update(_labels_from_node(info_semantic[key]))
    if not labels:
        labels.update(_labels_from_node(info_semantic))
    return sorted(label for label in labels if label not in CONTAMINANT_LABELS and label not in BANNED_BROAD_PROMPTS)


def build_replica_vocab(scene_info_paths: dict[str, Path]) -> dict[str, object]:
    """Build the global selected-scene Replica vocabulary from info_semantic files."""
    scene_class_coverage: dict[str, list[str]] = {}
    canonical_labels: set[str] = set()

    for scene_name, info_path in sorted(scene_info_paths.items()):
        payload = json.loads(Path(info_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {info_path}")
        scene_labels = _extract_class_names(payload)
        scene_class_coverage[str(scene_name)] = scene_labels
        canonical_labels.update(scene_labels)

    canonical_eval_vocab = sorted(canonical_labels)
    canonical_set = set(canonical_eval_vocab)
    banned_canonical = canonical_set & BANNED_BROAD_PROMPTS
    if banned_canonical:
        raise ValueError(f"Banned broad prompts in canonical_eval_vocab: {sorted(banned_canonical)}")
    structural_labels = sorted(canonical_set & STRUCTURAL_LABEL_CANDIDATES)
    structural_set = set(structural_labels)
    object_labels = [label for label in canonical_eval_vocab if label not in structural_set]
    prompt_aliases = {
        canonical: aliases
        for canonical, aliases in sorted(CONSERVATIVE_PROMPT_ALIASES.items())
        if canonical in canonical_set
    }
    banned_alias_keys = set(prompt_aliases) & BANNED_BROAD_PROMPTS
    if banned_alias_keys:
        raise ValueError(f"Banned broad prompts in prompt_aliases: {sorted(banned_alias_keys)}")
    banned_alias_values = {
        normalized_alias
        for aliases in prompt_aliases.values()
        for alias in aliases
        for normalized_alias in [_canonicalize_label(alias)]
        if normalized_alias in BANNED_BROAD_PROMPTS
    }
    if banned_alias_values:
        raise ValueError(f"Banned broad prompts in prompt_aliases: {sorted(banned_alias_values)}")

    return {
        "canonical_eval_vocab": canonical_eval_vocab,
        "prompt_aliases": prompt_aliases,
        "structural_labels": structural_labels,
        "object_labels": object_labels,
        "scene_class_coverage": scene_class_coverage,
    }


def default_scene_info_paths(replica_original_root: Path = DEFAULT_REPLICA_ORIGINAL_ROOT) -> dict[str, Path]:
    return {
        scene_name: Path(replica_original_root) / gt_scene / "habitat" / "info_semantic.json"
        for scene_name, gt_scene in SELECTED_SCENES.items()
    }


def build_summary(vocab: dict[str, object]) -> dict[str, object]:
    scene_class_coverage = dict(vocab.get("scene_class_coverage", {}) or {})
    return {
        "canonical_eval_vocab_count": len(vocab.get("canonical_eval_vocab", []) or []),
        "prompt_alias_count": len(vocab.get("prompt_aliases", {}) or {}),
        "structural_label_count": len(vocab.get("structural_labels", []) or []),
        "object_label_count": len(vocab.get("object_labels", []) or []),
        "scene_count": len(scene_class_coverage),
        "scene_class_counts": {
            str(scene): len(classes)
            for scene, classes in sorted(scene_class_coverage.items())
        },
        "scene_class_coverage": scene_class_coverage,
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-original-root", type=Path, default=DEFAULT_REPLICA_ORIGINAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_info_paths = default_scene_info_paths(args.replica_original_root)
    missing = [str(path) for path in scene_info_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Replica info_semantic.json files: " + ", ".join(missing))
    vocab = build_replica_vocab(scene_info_paths)
    _write_yaml(args.output, vocab)
    _write_json(args.summary_output, build_summary(vocab))
    print(
        "Wrote "
        f"{args.output} ({len(vocab['canonical_eval_vocab'])} classes, "
        f"{len(vocab['structural_labels'])} structural, "
        f"{len(vocab['object_labels'])} object)"
    )
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
