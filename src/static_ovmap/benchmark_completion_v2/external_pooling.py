"""Independent complete-method scene aggregation on the common official axis."""
from pathlib import Path

from .common import verified, write, source_identity


def ovi(binding):
    from .ovi_adapter import score
    from .standard_score import evaluator, ap_metrics
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    scopes = {}
    for benchmark, scope in protocol['scopes'].items():
        records, absent, matches = {}, [], {}
        for scene in scope['scenes']:
            nq = root / 'benchmark/NQ' / benchmark / scene / 'native_features.pkl'
            gt = root / 'benchmark/GT' / benchmark / scene / 'receipt.json'
            # Source availability is independent of the resulting metric.
            if not nq.is_file() or not gt.is_file():
                absent.append(scene)
                continue
            record = score(binding, benchmark, scene)
            evidence = verified(record['matches']['path'])
            if evidence['identity'] != record['matches']['identity']:
                raise ValueError('Complete-method match sufficient statistics changed')
            matches[scene] = evidence['matches'][scene]
            records[scene] = dict(score=record['identity'], prediction=record['prediction'],
                matches=evidence['identity'], native_model_source=verified(
                    root / 'benchmark/NQ' / benchmark / scene / 'sources.json')['identity'])
        metrics, classes = None, None
        if matches:
            metrics, classes, _ = ap_metrics(evaluator(binding, benchmark), matches)
        scopes[benchmark] = write(root / 'external/OVI-MAP/pools' / (benchmark + '.json'),
            dict(status='BLOCKED_DEPENDENCY' if absent else 'DONE', method='OVI-MAP', benchmark=benchmark,
                variant='ORIGIN_IMPLEMENTATION_STANDARD_PROTOCOL_PORT', locally_executed=True,
                metrics=metrics, per_class=classes, full_scene_coverage=not absent,
                completed_scenes=len(records), expected_scenes=len(scope['scenes']), missing_scenes=absent,
                ordered_scenes=list(records), source_scenes=records, protocol=protocol['identity'],
                scope='PARTIAL_SUBSET' if absent else 'FULL_STANDARD',
                aggregation='OFFICIAL_EVALUATE_MATCHES_ALL_ORDERED_SCENES;NO_SCENE_AP_AVERAGING',
                confidence='PRESERVED_AUTHOR_AREA_CURRENT_CLASS_RATIO_6_DECIMALS',
                new_GPU_encodings=0, new_training_updates=0, shared_original_N_only=True,
                official_pin='6304b5e93c14e66f25a01a64d3f650e0f1ccb0b9',
                producer=source_identity(('external_pooling',))))
    return write(root / 'external/OVI-MAP/pools/summary.json', dict(status='DONE' if all(
        r['full_scene_coverage'] for r in scopes.values()) else 'BLOCKED_DEPENDENCY', benchmarks=scopes))
