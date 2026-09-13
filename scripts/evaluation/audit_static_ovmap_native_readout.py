#!/usr/bin/env python3
"""Check every cached object against the original Torch readout arithmetic."""
import argparse
import ast
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import load_native_cache, sha256_file
from src.static_ovmap.readout import fuse_features, classify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-cache', type=Path, required=True)
    p.add_argument('--text-cache', type=Path, required=True)
    p.add_argument('--original-readout-source', type=Path, required=True)
    p.add_argument('--source-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tree = ast.parse(args.original_readout_source.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'match_feature_to_label_embed']
    if len(body) != 1:
        raise ValueError('original matching function missing')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(args.original_readout_source), 'exec'), ns)
    with args.native_cache.open('rb') as handle:
        native = pickle.load(handle)
    text = np.load(args.text_cache)
    space = str(text['feature_space_id'])
    bank, _ = load_native_cache(args.native_cache, scene_id='room0', feature_space_id=space,
        source_config_hash=sha256_file(args.source_config), history_scope='retained_native_top10')
    mismatches = []
    max_error = 0.
    eligible = 0
    for instance, record in native.items():
        if len(record['frame_id']) < 2:
            continue
        eligible += 1
        area = torch.tensor(record['vis_area'])[-8:]
        area = area / (area.sum() + 1e-6)
        feature = torch.sum(torch.tensor(record['feat'])[-8:] * area.unsqueeze(-1), dim=0)
        match = ns['match_feature_to_label_embed'](feature, torch.tensor(text['text']),
                    torch.tensor(text['canonical']), True)
        ours = classify(bank[int(instance)], text['text'], text['valid_ids'], space,
                         canonical_features=text['canonical'])
        if ours['class_id'] != int(text['valid_ids'][match]):
            mismatches.append(int(instance))
        error = float(np.max(np.abs(feature.numpy() - fuse_features(bank[int(instance)][-8:], 'vis_area'))))
        max_error = max(error, max_error)
    result = {'status': 'PASS' if not mismatches and max_error < 1e-6 else 'FAIL',
              'eligible_instances': eligible, 'label_mismatches': mismatches,
              'max_absolute_fused_feature_error': max_error,
              'feature_arithmetic': 'original_torch_float32_vs_numpy_float64_area_accumulation',
              'feature_tolerance': 1e-6, 'class_output_tolerance': 0,
              'source_sha256': sha256_file(args.original_readout_source), 'command': sys.argv}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
