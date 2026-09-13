#!/usr/bin/env python3
"""Classify declared native components from existing native B0 queries only."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import load_native_cache, validate_native_binding, sha256_file
from src.static_ovmap.hierarchy import component_readout
from src.static_ovmap.readout import classify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--split', type=Path)
    p.add_argument('--geometry-support', type=Path, required=True)
    p.add_argument('--native-cache', type=Path, required=True)
    p.add_argument('--native-binding', type=Path, required=True)
    p.add_argument('--source-config', type=Path, required=True)
    p.add_argument('--text-cache', type=Path, required=True)
    p.add_argument('--scene', default='room0')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    graph = json.loads(args.graph.read_text())
    geometry = json.loads(args.geometry_support.read_text())
    if (geometry['native_cache_sha256'] != sha256_file(args.native_cache)
            or graph['input_sha256'].get(str(args.geometry_support)) != sha256_file(args.geometry_support)):
        raise ValueError('graph owner identity and native semantic cache are not bound')
    text = np.load(args.text_cache, allow_pickle=False)
    space = str(text['feature_space_id'])
    bank, metadata = load_native_cache(args.native_cache, scene_id=args.scene, feature_space_id=space,
        source_config_hash=sha256_file(args.source_config), history_scope='retained_native_top10')
    validate_native_binding(json.loads(args.native_binding.read_text()), metadata['source_sha256'],
        sha256_file(args.source_config), space, text['text'].shape[1])
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    baseline = {str(owner): classify(obs, text['text'], text['valid_ids'], space,
                                    canonical_features=text['canonical']) for owner, obs in bank.items()}
    b0_seconds = time.perf_counter()-start
    start = time.perf_counter()
    merged = component_readout(graph['partition']['components'], bank, baseline,
                               text['text'], text['valid_ids'], space, text['canonical'])
    g1_seconds = time.perf_counter()-start
    conditions = [('B0', {'observations': baseline}, b0_seconds), ('G1', merged, g1_seconds)]
    if args.split:
        split = json.loads(args.split.read_text())
        if split['input_sha256'].get(str(args.graph)) != sha256_file(args.graph):
            raise ValueError('split and G1 graph identity mismatch')
        start = time.perf_counter()
        native_g2 = np.load(args.split.parent/'G2_native_owners.npy', allow_pickle=False)
        observations = {}
        for owner in np.unique(native_g2[native_g2 > 0]):
            parent = split['child_sources'].get(str(owner), int(owner))
            observations[str(owner)] = merged['observations'].get(str(parent))
        conditions.append(('G2', {'observations': observations,
            'semantic_rule': 'children_inherit_G1_parent_no_new_queries',
            'split_sha256': sha256_file(args.split)}, time.perf_counter()-start))
    for condition, readout, seconds in conditions:
        document = {'condition': condition, 'scene': args.scene, **readout,
                    'mode': 'STATIC_OFFLINE_NATIVE_GRAPH_READOUT', 'readout_seconds': seconds,
                    'image_encoder_calls': 0, 'feature_space_id': space,
                    'history_scope': 'retained_native_top10', 'graph_sha256': sha256_file(args.graph)}
        (args.output/f'{condition}.json').write_text(json.dumps(document, indent=2)+'\n')
    inputs = [args.graph, args.geometry_support, args.native_cache,
              args.native_binding, args.source_config, args.text_cache]
    if args.split:
        inputs.append(args.split)
    (args.output/'input_binding.json').write_text(json.dumps({'command': sys.argv, 'native_cache': metadata,
        'input_sha256': {str(f):sha256_file(f) for f in inputs}}, indent=2)+'\n')
    print('B0', b0_seconds, 'G1', g1_seconds, flush=True)


if __name__ == '__main__':
    main()
