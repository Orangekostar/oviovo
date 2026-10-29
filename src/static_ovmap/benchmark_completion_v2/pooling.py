"""Aggregate only frozen per-scene official matches, never scene AP averages."""
from pathlib import Path

from .common import verified, write, canonical_digest, source_identity


def standard(binding, benchmark):
    from .standard_score import evaluator, ap_metrics
    root = Path(binding['output_root'])
    scope = verified(root / 'protocol_lock.json')['scopes'][benchmark]
    scenes = scope['scenes']
    available = {}
    for scene in scenes:
        path = root / 'benchmark/evaluation' / benchmark / scene / 'receipt.json'
        if path.is_file():
            available[scene] = verified(path)
    namespace = evaluator(binding, benchmark)
    rows, missing = {}, []
    for method in ('U00', 'U10', 'U01', 'U11'):
        for seed in (17, 29):
            key = method + '_seed' + str(seed)
            ordered = [(scene, available[scene]['rows'][key]) for scene in scenes
                       if scene in available and key in available[scene]['rows']]
            absent = [scene for scene in scenes if scene not in available or key not in available[scene]['rows']]
            if not ordered:
                missing.append(dict(method=key, scenes=absent))
                continue
            matches, inputs = {}, []
            for scene, row in ordered:
                match = verified(row['matches']['path'])
                if match['identity'] != row['matches']['identity']:
                    raise ValueError('Frozen official match cache changed before pooling')
                matches[scene] = match['matches'][scene]
                inputs.append(dict(scene=scene, prediction=row['prediction_identity'], matches=match['identity']))
            metrics, classes, _ = ap_metrics(namespace, matches)
            identity = canonical_digest(dict(inputs=inputs, protocol=scope,
                producer=source_identity(('pooling',))))
            row = write(root / 'benchmark/pools' / benchmark / (key + '.json'),
                dict(status='DONE' if not absent else 'BLOCKED_DEPENDENCY', input_identity=identity,
                    method=method, seed=seed, benchmark=benchmark, metrics=metrics, per_class=classes,
                    ordered_scenes=[s for s, _ in ordered], expected_scenes=scenes, missing_scenes=absent,
                    full_scene_coverage=not absent, scope='FULL_STANDARD' if not absent else 'PARTIAL_SUBSET',
                    aggregation='OFFICIAL_EVALUATE_MATCHES_ALL_ORDERED_SCENES',
                    no_scene_AP_averaging=True, metric_unit='FRACTION', inputs=inputs))
            rows[key] = row
            if absent:
                missing.append(dict(method=key, scenes=absent))
    return write(root / 'benchmark/pools' / benchmark / 'summary.json', dict(
        status='DONE' if not missing else 'BLOCKED_DEPENDENCY', benchmark=benchmark, rows=rows,
        expected_pools=8, completed_full_pools=sum(r['full_scene_coverage'] for r in rows.values()),
        partial_subset_pools=sum(not r['full_scene_coverage'] for r in rows.values()), missing=missing))
