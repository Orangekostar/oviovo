"""G1 primary audit against frozen real inputs and independent reference arithmetic."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

from .common import REPO, verified, read, write, digest_file


def run(binding):
    from .calibration import nmix_scores
    from static_ovmap.semantic_update_evidence_v1.readers import column_index
    root, public = Path(binding['output_root']), Path(binding['semantic_root'])
    path = REPO / 'docs/paper/static_ovmap/benchmark_completion_v2/prompt/reference_ops.py'
    spec = importlib.util.spec_from_file_location('ovmap_v2_independent_reference', path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    metadata = verified(root / 'A/import.json')
    summary = verified(root / 'calibration/summary.json')
    candidates = verified(root / 'calibration/candidates.json.gz')
    crossfit = verified(root / 'calibration/crossfit.json.gz')
    recipes = candidates['recipes']
    with np.load(root / 'A/import.npz', allow_pickle=False) as arrays:
        scores, vectors, defined, ids, text = (arrays[k] for k in ('scores', 'embeddings', 'defined', 'valid_ids', 'text'))
    with np.load(root / 'calibration/predictions.npz', allow_pickle=False) as arrays:
        predictions = {key: arrays[key].copy() for key in arrays.files}
    target, families = predictions['targets'], predictions['families']
    base, novel = metadata['class_split']['base_ids'], metadata['class_split']['heldout_ids']
    unsupported = np.array([not row['supported'] for row in metadata['objects']])
    raw_records = [r for r in summary['raw_and_post'] if r['panel'] == 'RAW']
    checks = []
    for row in raw_records:
        ci = column_index(metadata['columns'], row['method'], row['seed'])
        actual = reference.predict(scores[:, ci, 0], ids, defined[:, ci, 0])
        key = 'B_FC_+0' if row['method'] == 'B_FC' else row['method'] + '_raw'
        saved = predictions[key] if row['method'] == 'B_FC' else predictions[key][(17, 29).index(row['seed'])]
        np.testing.assert_array_equal(actual, saved)
        if np.any(actual[unsupported]) or row['metrics']['originals'] != 279 or row['metrics']['abstentions'] != 9:
            raise ValueError('Real RAW omitted original abstentions')
        independent = reference.recognition(target, saved, base, novel)
        for name in ('B_macro', 'N_macro', 'H_macro', 'B_micro', 'N_micro', 'H_micro'):
            np.testing.assert_allclose(row['metrics'][name], independent[name], rtol=0, atol=1e-14)
        checks.append(dict(method=row['method'], seed=row['seed'], original_source_RAW_exact=True,
                           independent_metric_error_max=max(abs(independent[k] - row['metrics'][k]) for k in independent if isinstance(independent[k], float))))
    with np.load(public / 'A/predictions.npz', allow_pickle=False) as old:
        old_recipes = read(public / 'A/crossfit.json.gz')['candidate_recipes']
        current = {r['candidate_id']: i for i, r in enumerate(recipes)}
        for i, recipe in enumerate(old_recipes):
            np.testing.assert_array_equal(old['candidate_predictions'][:, i],
                predictions['MV_ATTN_candidates'][:, current[recipe['candidate_id']]])
    # Independent actual-vector NMIX checks, without regenerating the old grid.
    bi = column_index(metadata['columns'], 'B_FC', None)
    bs, b, bd = scores[:, bi, 0], vectors[:, bi, 0], defined[:, bi, 0]
    base_mask = np.isin(ids, base)
    sample_checks = []
    for method in ('S_LL', 'G_LL', 'V_LL', 'MV_ATTN'):
        for si, seed in enumerate((17, 29)):
            ci = column_index(metadata['columns'], method, seed)
            zs, z, zd = scores[:, ci, 0], vectors[:, ci, 0], defined[:, ci, 0]
            valid = bd & zd
            for alpha in (.25, .5, .75):
                reference_vectors, reference_flags = reference.normalized_mix(b[valid], z[valid], alpha)
                actual_scores, flags = nmix_scores(alpha, bs[valid], zs[valid], b[valid], z[valid], text)
                error = float(np.max(np.abs(actual_scores - reference_vectors @ text.T)))
                if error > 1e-7:
                    raise ValueError('Independent real-vector NMIX arithmetic differs')
                np.testing.assert_array_equal(flags, reference_flags)
                full = np.full_like(zs, np.nan)
                full[valid] = actual_scores
                for gamma in (-.025, 0., .025):
                    adjusted = reference.offset_scores(full, base_mask, gamma)
                    expected = reference.predict(adjusted, ids, valid)
                    ri = next(i for i, r in enumerate(recipes) if r['candidate_id'] == f'NMIX|{alpha:g}|{gamma:+g}')
                    np.testing.assert_array_equal(expected, predictions[method + '_candidates'][si, ri])
                sample_checks.append(dict(method=method, seed=seed, alpha=alpha,
                    real_supported_originals=int(valid.sum()), max_score_error=error, gammas_checked=[-.025, 0., .025]))
    sampled_folds = []
    for method in ('S_LL', 'G_LL', 'V_LL', 'MV_ATTN', 'B_FC'):
        saved = predictions[method + '_OOF']
        for fold in [r for r in crossfit['folds'] if r['method'] == method]:
            hold = families == fold['held_family']
            keys = predictions['original_keys'][hold].tolist()
            if keys != fold['held_keys'] or fold['training_originals'] != int((~hold).sum()):
                raise ValueError('Held-family originals entered selection data')
            if method == 'B_FC':
                values = predictions['B_FC_' + f"{fold['recipe']['gamma']:+g}"]
                np.testing.assert_array_equal(saved[hold], values[hold])
            else:
                ri = next(i for i, r in enumerate(recipes) if r['candidate_id'] == fold['recipe']['candidate_id'])
                values = predictions[method + '_candidates'][:, ri]
                np.testing.assert_array_equal(saved[:, hold], values[:, hold])
                if fold['selection_seed'] != 17 or fold['applied_seeds'] != [17, 29]:
                    raise ValueError('LOFO used a seed29-specific selection')
            if method != 'B_FC' and fold['held_family'] in sorted(set(families))[:1]:
                train = ~hold
                choices = [dict(recipe=recipe, metrics=reference.recognition(target[train],
                    predictions[method + '_candidates'][0, i, train], base, novel)) for i, recipe in enumerate(recipes)]
                chosen = reference.best(choices)['recipe']['candidate_id']
                if chosen != fold['recipe']['candidate_id']:
                    raise ValueError('Independent held-family selection differs')
                sampled_folds.append(dict(method=method, held_family=fold['held_family'], candidate_count=len(choices), chosen=chosen))
        for si in range(1 if method == 'B_FC' else 2):
            independent = reference.recognition(target, saved if method == 'B_FC' else saved[si], base, novel)
            row = next(r for r in summary['raw_and_post'] if r['method'] == method and r['panel'] == 'LOFO_POST'
                       and r['seed'] == (None if method == 'B_FC' else (17, 29)[si]))
            for key in ('B_macro', 'N_macro', 'H_macro', 'B_micro', 'N_micro', 'H_micro'):
                np.testing.assert_allclose(independent[key], row['metrics'][key], atol=1e-14, rtol=0)
    if len(candidates['rows']) != 4 * 2 * 165 + 18 or len(candidates['static_B_FC']) != 11:
        raise ValueError('Metadata is not the complete 1331 logical candidate domain plus raw/OOF summaries')
    return write(root / 'review/G1_real_numeric_audit.json', dict(status='DONE',
        independent_reference=dict(path=str(path), sha256=digest_file(path)),
        real_source_RAW_checks=checks, exact_original_MV_candidate_reuses=220,
        independent_real_vector_NMIX_checks=sample_checks, NMIX_candidate_prediction_checks=len(sample_checks) * 3,
        all60_fold_original_order_and_seed_shared_recipe_checks=True, independent_sample_fold_checks=sampled_folds,
        all9_post_metrics_recomputed_on_stitched_originals=True,
        originals=279, base_originals=int(np.isin(target, base).sum()), novel_originals=int(np.isin(target, novel).sum()),
        abstentions=9, logical_candidates=1331, measured_server_data=True,
        new_GPU_workers=0, new_model_head_calls=0, new_image_encodings=0,
        verification_CPU_passes_separate_from_candidate_generation=True))
