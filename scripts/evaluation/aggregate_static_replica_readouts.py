#!/usr/bin/env python3
"""Aggregate complete Replica8 readouts using released pooled AP and vertex confusion."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import fmean
import sys

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.evaluate_static_ovmap_instances import finite_json
from scripts.evaluation.run_static_ovmap_readout import CONDITIONS as READOUT_RULES
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.pooled_metrics import pooled_semantic_metrics
from src.static_ovmap.released_loader import load_released_module

SCENES = ('office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2')
CONDITIONS = ('B0', 'S1a', 'S1b', 'S1c', 'C1', 'RANDOM8', 'QUALITY8', 'ALL_VIEWS',
              'S1a_FULL', 'RANDOM8_FULL', 'QUALITY8_FULL', 'ALL_VIEWS_FULL')


def complete_mean(values):
    values = list(values)
    return fmean(values) if values and all(v is not None for v in values) else None


def aggregate_group(records, conditions, evaluator_root):
    """Also used for explicitly labelled subset integration checks, never implied Replica8."""
    evaluator = load_released_module(Path(evaluator_root)/'scripts/eval_utils.py')
    evaluator['init']('Replica')
    gt_paths = [r['gt_ids'] for r in records.values()]
    if len({str(p.resolve()) for p in gt_paths}) != len(records):
        raise ValueError('each scene needs a distinct GT instance file')
    ground_truth = {s: np.asarray(PlyData.read(r['gt_semantic'])['vertex']['label']) for s, r in records.items()}
    pred_root = os.path.commonpath([str(r['instances'].resolve()) for r in records.values()])
    rows = {}
    for condition in conditions:
        pairs = [(ground_truth[s], np.load(r['semantic']/(condition+'_semantic_labels.npy'), allow_pickle=False))
                 for s, r in records.items()]
        semantic = pooled_semantic_metrics(pairs, list(range(1, 52)))
        pred_files = [str(r['instances']/condition/'pred_inst_sem_mapping.txt') for r in records.values()]
        ap = finite_json(evaluator['evaluate'](pred_root, pred_files, list(map(str, gt_paths)), pred_root))
        local_sem = [json.loads((r['semantic']/(condition+'_metrics.json')).read_text()) for r in records.values()]
        local_ap = [json.loads((r['instances']/condition/'metrics.json').read_text())['released_semantic_instance']
                    for r in records.values()]
        rows[condition] = {'pooled_semantic': semantic, 'pooled_released_semantic_instance': ap,
            'scene_macro_diagnostic': {'semantic_miou': complete_mean(x['semantic_miou'] for x in local_sem),
                'semantic_macc': complete_mean(x['semantic_macc'] for x in local_sem),
                'semantic_instance_AP': complete_mean(x['all_ap'] for x in local_ap),
                'missing_policy': 'null if any scene metric is undefined; no zero imputation'},
            'null_class_ap_reason': 'released evaluator has no GT instances for that class in this subset'}
    return {'scene_ids': list(records), 'scene_count': len(records), 'conditions': rows,
            'baseline_role': 'internal B0 is rebuilt native B1, not the historical Room0 cache',
            'AP_definition': 'one released evaluate call across all scene manifests per condition; not mean scene AP',
            'semantic_definition': 'pooled vertex confusion before averaging present GT classes'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-result', action='append', required=True, help='SCENE=completed scene pipeline directory')
    p.add_argument('--evaluator-root', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    directories = {}
    for raw in args.scene_result:
        scene, path = raw.split('=', 1)
        if scene in directories:
            raise ValueError('duplicate scene: '+scene)
        directories[scene] = Path(path)
    if set(directories) != set(SCENES):
        raise ValueError('exactly the fixed eight Replica scenes are required')
    records, sources, signatures, costs, diagnostics, per_scene = {}, {}, set(), {}, {}, {}
    for scene in SCENES:
        root = directories[scene]
        job = json.loads((root/'scene_pipeline.json').read_text())
        if job.get('status') != 'COMPLETE' or job.get('scene') != scene or job['frame_ids'] != list(range(0, 2000, 10)):
            raise ValueError('completed, correctly bound scene pipeline required: '+scene)
        if sha256_file(job['native_manifest']) != job['native_manifest_sha256']:
            raise ValueError('native manifest changed: '+scene)
        if json.loads((root/'projection_audit/projection_audit.json').read_text())['status'] != 'PASS':
            raise ValueError('independent projection audit missing: '+scene)
        metadata = json.loads((root/'readout/input_binding.json').read_text())
        signatures.add((metadata['source_config_sha256'], metadata['text_cache_sha256'],
                        metadata['text_metadata']['feature_space_id']))
        records[scene] = {'semantic': root/'semantic', 'instances': root/'instances',
                         'gt_ids': root/'instances/gt_sem_inst_id.npy',
                         'gt_semantic': root/'layout'/scene/'gt_semantic_mesh.ply'}
        instance_summary = json.loads((root/'instances/instance_summary.json').read_text())
        if instance_summary['source_sha256']['eval_utils.py'] != sha256_file(args.evaluator_root/'scripts/eval_utils.py'):
            raise ValueError('released evaluator source differs from scene evaluation')
        local = {r['condition']: r for r in instance_summary['rows']}
        baseline = json.loads((root/'readout/B0.json').read_text())['observations']
        bridge = metadata.get('full_history', {}).get('native_query_to_full_query', {})
        source_files = [root/'scene_pipeline.json', root/'readout/input_binding.json', root/'instances/instance_summary.json',
                        root/'geometry/metrics.json', root/'projection_audit/projection_audit.json']
        diagnostics[scene] = {}
        per_scene[scene] = {}
        for condition in args.conditions:
            readout_path = root/'readout'/(condition+'.json')
            doc = json.loads(readout_path.read_text())
            if doc['scene'] != scene or doc['seed'] != 0 or doc['mode'] != 'STATIC_OFFLINE_READOUT':
                raise ValueError('scene/seed/offline condition differs')
            if condition.endswith('_FULL') and not job['include_full_history']:
                raise ValueError('full-history result without complete capture')
            for observation in doc['observations'].values():
                if observation is not None and (observation['strategy'], observation['weighting']) != READOUT_RULES[condition]:
                    raise ValueError('selector/fusion condition differs from the frozen configuration')
            manifest = root/'instances'/condition/'pred_inst_sem_mapping.txt'
            if sha256_file(manifest) != local[condition]['prediction_manifest_sha256']:
                raise ValueError('prediction manifest changed')
            changes, subsets = [], 0
            for owner, before in baseline.items():
                after = doc['observations'][owner]
                if before is None or after is None:
                    if before != after:
                        raise ValueError('pure selector changed native eligibility')
                    continue
                normalize = lambda ids: {bridge.get(q, q) for q in ids}
                subsets += normalize(before['selected_query_ids']) != normalize(after['selected_query_ids'])
                if before['class_id'] != after['class_id']:
                    changes.append({'owner': owner, 'before': before, 'after': after})
            diagnostics[scene][condition] = {'changed_selection_sets': subsets, 'label_changes': changes,
                'readout_seconds': doc['readout_seconds'], 'history_scope': doc['history_scope'],
                'query_budget': doc['query_budget']}
            semantic = json.loads((root/'semantic'/(condition+'_metrics.json')).read_text())
            ap = local[condition]['released_semantic_instance']
            per_scene[scene][condition] = {'semantic_miou': semantic['semantic_miou'],
                'semantic_macc': semantic['semantic_macc'], 'released_semantic_AP': ap['all_ap'],
                'released_semantic_AP50': ap['all_ap_50%'], 'released_semantic_AP25': ap['all_ap_25%'],
                'released_semantic_filtered_geometry': local[condition]['released_class_agnostic'],
                'full_native_geometry': json.loads((root/'geometry/metrics.json').read_text())['canonical_class_agnostic'],
                'history_scope': doc['history_scope'], 'query_budget': doc['query_budget']}
            source_files += [readout_path, manifest, root/'semantic'/(condition+'_semantic_labels.npy')]
        sources[scene] = {str(f): {'sha256': sha256_file(f), 'size_bytes': f.stat().st_size} for f in source_files}
        native = json.loads(Path(job['native_manifest']).read_text())
        stages = {}
        for name, r in [('frontend', native['frontend']), ('geometry', native['mapping']['geometry']),
                        ('mapping_and_vlm_and_export', native['mapping']['mapping'])]:
            elapsed = (datetime.fromisoformat(r['finished_at'])-datetime.fromisoformat(r['started_at'])).total_seconds()
            stages[name] = {'original_recorded_wall_seconds': elapsed, 'reused': r.get('reused', False),
                            'incremental_wall_seconds': 0 if r.get('reused', False) else elapsed}
        costs[scene] = {'native_stages': stages, 'readout_and_evaluation_stages': job['stages'],
            'peak_gpu_bytes': None, 'isolated_vlm_seconds': None, 'isolated_export_seconds': None,
            'missing_reason': 'not independently measured by native runner',
            'cost_boundary': 'concurrent shared host; reused original timestamps are not new work; no isolated E2E FPS'}
    if len(signatures) != 1:
        raise ValueError('native config/text/feature-space identity differs across scenes')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, subset in [('replica8', records), ('replica7_diagnostic', {s:r for s,r in records.items() if s != 'room0'})]:
        group = aggregate_group(subset, args.conditions, args.evaluator_root)
        geometry = [json.loads((directories[s]/'geometry/metrics.json').read_text()) for s in subset]
        group['canonical_geometry_scene_macro_diagnostic'] = {
            k: complete_mean(g['canonical_class_agnostic'][k] for g in geometry) for k in ('ap25', 'ap50', 'ap75')}
        group['paper_AP'] = None
        group['paper_AP_reason'] = 'paper class-agnostic AP provenance not verified; canonical macro is separate'
        group['development_boundary'] = 'Room0 used for development; extra seven scenes are not claimed as a new independent test set'
        write(args.output/(name+'.json'), group)
    write(args.output/'selection_diagnostics.json', diagnostics)
    write(args.output/'per_scene.json', per_scene)
    write(args.output/'costs.json', costs)
    write(args.output/'source_index.json', {'command': sys.argv, 'sources': sources,
        'aggregator_sha256': sha256_file(__file__),
        'metric_source_sha256': {str(f): sha256_file(f) for f in
            (ROOT/'src/static_ovmap/pooled_metrics.py', args.evaluator_root/'scripts/eval_utils.py')}})
    print(args.output)


if __name__ == '__main__':
    main()
