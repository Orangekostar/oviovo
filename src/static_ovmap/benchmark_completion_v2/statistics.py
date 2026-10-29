"""Only the two contracted paired family bootstraps; pooled AP for each seed."""
import copy
from pathlib import Path
import numpy as np

from .common import SEEDS, verified, write, canonical_digest, source_identity, StageFailure


def namespaced(matches, name):
    """Namespace every occurrence, including the references inside GT matches."""
    def visit(value):
        if isinstance(value, dict):
            return {key: (name + ':' + str(item) if key in ('uuid', 'filename') else visit(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value
    return visit(matches)


def paired_bootstrap(binding, benchmark, *, draws=1000):
    from .standard_score import evaluator, ap_metrics
    contract=binding['specification']['statistics']
    if draws != 1000 or draws != contract['bootstrap_draws']:
        raise ValueError('Frozen scientific bootstrap requires exactly1000 shared family draws')
    random_seed=int(contract['seed'])
    root = Path(binding['output_root'])
    scope = verified(root / 'protocol_lock.json')['scopes'][benchmark]
    scenes = scope['scenes']
    records, missing = {}, []
    for scene in scenes:
        path = root / 'benchmark/evaluation' / benchmark / scene / 'receipt.json'
        if not path.is_file():
            missing.append(str(path))
            continue
        receipt = verified(path)
        for seed in SEEDS:
            for method in ('U11', 'U10', 'U01'):
                key = method + '_seed' + str(seed)
                row = receipt['rows'].get(key)
                if row is None:
                    missing.append(str(path) + '#' + key)
                    continue
                descriptor = row['matches']
                record = verified(descriptor['path'])
                if record['identity'] != descriptor['identity']:
                    raise ValueError('Bootstrap match cache changed')
                records[scene, method, seed] = record['matches'][scene]
    if missing:
        raise StageFailure('Full-scene shared paired bootstrap needs the complete actual U grid', missing=missing)
    identity = canonical_digest(dict(benchmark=benchmark, protocol_scope=scope,
        records={str(k): v for k, v in records.items()},
        draws=draws,statistics_contract=contract,producer=source_identity(('statistics',))))
    path = root / 'statistics' / (benchmark + '.json')
    if path.exists():
        prior = verified(path)
        if prior['input_identity'] != identity:
            raise ValueError('Completed paired-bootstrap inputs changed')
        return prior
    family_scenes = {family: [s for s in scenes if (s.split('_')[0] if s.startswith('scene') else s) == family]
                     for family in scope['families']}
    families = list(family_scenes)
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(0, len(families), size=(draws, len(families)))
    cache = root / 'statistics/checkpoints' / benchmark
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        if verified(manifest_path)['input_identity'] != identity:
            raise ValueError('Pending paired-bootstrap input identity changed')
    else:
        write(manifest_path, dict(status='DONE', scope='BOOTSTRAP_DRAW_CACHE_INPUT_LOCK_ONLY',
            input_identity=identity, benchmark=benchmark, draws=draws, random_seed=random_seed,
            families=families, draw_plan_sha256=canonical_digest(sample_indices)))
    namespace = evaluator(binding, benchmark)
    results = {other: [] for other in ('U10', 'U01')}
    seed_values = {other: [] for other in results}
    for di, sample in enumerate(sample_indices):
        draw_path = cache / f'draw_{di:04d}.json'
        if draw_path.exists():
            saved = verified(draw_path)
            if (saved['input_identity'] != identity or saved['draw_index'] != di
                    or saved['family_indices'] != sample.tolist()):
                raise ValueError('Cached paired-bootstrap draw identity changed')
            aps = {(method, seed): saved['pooled_APall'][method + '_seed' + str(seed)]
                   for method in ('U11', 'U10', 'U01') for seed in SEEDS}
        else:
            methods = {(method, seed): {} for method in ('U11', 'U10', 'U01') for seed in SEEDS}
            for occurrence, fi in enumerate(sample):
                for scene in family_scenes[families[int(fi)]]:
                    tag = f'draw{di}:occurrence{occurrence}:{scene}'
                    for (method, seed), matches in methods.items():
                        matches[tag] = namespaced(records[scene, method, seed], tag)
            aps = {key: ap_metrics(namespace, matches)[0]['apall'] for key, matches in methods.items()}
            write(draw_path, dict(status='DONE', input_identity=identity, draw_index=di,
                family_indices=sample.tolist(), pooled_APall={method + '_seed' + str(seed): value
                for (method, seed), value in aps.items()},
                scope='ONE_COMPLETE_SHARED_FAMILY_DRAW_ALL_THREE_METHODS_BOTH_SEEDS'))
        for other in results:
            differences = [None if aps['U11', seed] is None or aps[other, seed] is None else
                           aps['U11', seed] - aps[other, seed] for seed in SEEDS]
            seed_values[other].append(differences)
            results[other].append(None if any(value is None for value in differences) else float(np.mean(differences)))
        if di % 25 == 0:
            print('BOOTSTRAP', benchmark, di + 1, '/', draws, flush=True)
    summaries = {}
    for other, values in results.items():
        valid = [v for v in values if v is not None]
        summaries['U11-' + other] = dict(draws=draws, valid_draws=len(valid), undefined_draws=draws-len(valid),
            percentile_95=None if not valid else np.quantile(valid, [.025, .975]).tolist(),
            mean_delta=None if not valid else float(np.mean(valid)), mean_delta_draws=values,
            per_seed_deltas=seed_values[other], metric='POOLED_APall_FRACTION')
    return write(path, dict(status='DONE', input_identity=identity, benchmark=benchmark,
        scenes=scenes, families=families, shared_family_draws=sample_indices,
        comparisons=summaries, final_seed_differences_averaged=True,
        duplicate_scene_and_prediction_namespaces=True, undefined_draws_resampled=False,
        scientific_draw_sets=2, random_seed=random_seed,statistics_contract=contract))
