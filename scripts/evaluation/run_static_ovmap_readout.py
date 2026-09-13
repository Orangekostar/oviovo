#!/usr/bin/env python3
"""Read retained native observations without opening any evaluation/GT input."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import load_native_cache, sha256_file, validate_native_binding
from src.static_ovmap.readout import classify
from src.static_ovmap.enrichment import apply_enrichment


CONDITIONS = {'B0': ('last8', 'vis_area'), 'RANDOM8': ('random8', 'vis_area'),
              'QUALITY8': ('quality8', 'vis_area'),
              'S1a': ('quality_coverage8', 'vis_area'),
              'S1b': ('last8', 'quality'), 'S1c': ('quality_coverage8', 'quality'),
              'ALL_VIEWS': ('all_views', 'vis_area')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-cache', type=Path, required=True)
    parser.add_argument('--native-binding', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--history-scope', required=True)
    parser.add_argument('--enrichment', type=Path)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    text = np.load(args.text_cache, allow_pickle=False)
    space = str(text['feature_space_id'])
    binding = json.loads(args.native_binding.read_text())
    bank, metadata = load_native_cache(args.native_cache, scene_id=args.scene,
        feature_space_id=space, source_config_hash=sha256_file(args.source_config),
        history_scope=args.history_scope)
    validate_native_binding(binding, metadata['source_sha256'], sha256_file(args.source_config),
                            space, int(text['text'].shape[1]))
    if any(obs.feature_dim != binding['feature_dim'] for observations in bank.values() for obs in observations):
        raise ValueError('native record feature dimension contradicts binding')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata.update({'text_cache_sha256': sha256_file(args.text_cache),
                     'text_metadata': json.loads(args.text_cache.with_suffix('.json').read_text()),
                     'source_config_sha256': sha256_file(args.source_config),
                     'source_config_path': str(args.source_config.resolve()),
                     'quality_mode': 'area_only_no_quality_measurements',
                     'direction_mode': 'missing_no_object_relative_geometry',
                     'additional_image_queries': 0, 'command': sys.argv})
    metadata['native_binding_path'] = str(args.native_binding.resolve())
    metadata['native_binding_sha256'] = sha256_file(args.native_binding)
    if args.enrichment:
        enrichment = json.loads(args.enrichment.read_text())
        bank = apply_enrichment(bank, enrichment, metadata['source_sha256'])
        metadata.update({'enrichment_path': str(args.enrichment.resolve()),
                         'enrichment_sha256': sha256_file(args.enrichment),
                         'quality_mode': enrichment['quality_mode'],
                         'direction_mode': enrichment['direction_mode'],
                         'enrichment_preparation_seconds': enrichment['total_preparation_seconds']})
    (args.output / 'input_binding.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for condition in args.conditions:
        strategy, weighting = CONDITIONS[condition]
        start = time.perf_counter()
        result = {str(k): classify(obs, text['text'], text['valid_ids'], space,
                   strategy=strategy, weighting=weighting, seed=args.seed,
                   canonical_features=text['canonical']) for k, obs in bank.items()}
        seconds = time.perf_counter() - start
        document = {'condition': condition, 'scene': args.scene, 'seed': args.seed,
                    'observations': result, 'readout_seconds': seconds,
                    'mode': 'STATIC_OFFLINE_READOUT', 'history_scope': args.history_scope,
                    'query_budget': 'all_retained' if strategy == 'all_views' else 8,
                    'quality_mode': metadata['quality_mode'], 'direction_mode': metadata['direction_mode'],
                    'cache_scope': 'warm_image_and_text_features', 'peak_gpu_memory_bytes': 0,
                    'frontend_seconds': None, 'vlm_image_seconds': None,
                    'end_to_end_seconds': None,
                    'cost_missing_reason': 'historical_frontend_reused_not_timed_this_run'}
        (args.output / (condition+'.json')).write_text(json.dumps(document, indent=2)+'\n')
        print(condition, 'eligible_instances', sum(v is not None for v in result.values()),
              'readout_seconds', round(seconds, 6))


if __name__ == '__main__':
    main()
