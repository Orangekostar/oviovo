"""One requirement-to-evidence audit, including real portable reconstruction."""
import hashlib
from pathlib import Path
import time

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.protocol import validate_overlaps
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.evaluation import scorer_context

from .binding import load_scene
from .bundles import restore_payload, unpack_json
from .common import REPO, ConsumptionIndex, _array_digest, canonical_digest, read, verified, write
from .policies import anchor_interest, score_bank
from .query_plan import select_targets
from .selection import choose_screen, screen_gate
from .updates import Evidence, update


def verify(binding):
    root = Path(binding['output_root']); artifacts = REPO/'artifacts/static_ovmap/disagreement_query_v1'
    screen = verified(root/'screen_store.json'); store = verified(root/'result_store.json')
    selection = verified(root/'screen_selection.json'); regression = verified(root/'regression.json')
    timing = verified(root/'timing.json'); reports = verified(root/'reporting.json')
    parity = verified(root/'verification/original_scorer_parity.json')
    bundle = verified(artifacts/'bundles/manifest.json'); bundle_root = artifacts/'bundles'
    if not (screen['scene_method_coverage'] == 40 and screen['ordered_pool_coverage'] == 20):
        raise ValueError('real screen coverage incomplete')
    index = ConsumptionIndex(root/'verification/input_verifications.json')
    for freeze in ('query_kernel_freeze.json', 'screen_prediction_freeze.json'):
        f = verified(root/freeze)
        for record in f['operators']: index.identity(record['path'], record)
        index.identity((f.get('spec') or f.get('numerical_spec'))['path'], f.get('spec') or f['numerical_spec'])
    for file in bundle['files'].values():
        path = bundle_root/file['path']; raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != file['stored_sha256']:
            raise ValueError('portable artifact bytes differ')
        if file['encoding'] == 'GZIP_JSON':
            import gzip
            raw = gzip.decompress(raw)
        if hashlib.sha256(raw).hexdigest() != file['source_sha256']:
            raise ValueError('portable original bytes do not restore')
    registry_path = bundle_root/bundle['payload_registries']['path']
    if hashlib.sha256(registry_path.read_bytes()).hexdigest() != bundle['payload_registries']['sha256']:
        raise ValueError('portable payload registry changed')
    registries = unpack_json(registry_path)
    scenes = [s for group in binding['screen_cohorts'].values() for s in group]
    restored_count = posterior_count = policies_count = 0
    engineering_objects = {r['scene']: r['owner'] for r in verified(root/'engineering/receipt.json')['objects']}
    fixed_set_checks = []
    aggregation_costs = {}
    for scene in scenes:
        predictor, inputs = load_scene(binding, scene)
        if {'nearest', 'matched', 'targets', 'units', 'data'}.intersection(vars(predictor)):
            raise ValueError('actual predictor namespace contains evaluation data')
        scene_bundle = bundle['scenes'][scene]
        plan = unpack_json(bundle_root/scene_bundle['plan'])
        choices = unpack_json(bundle_root/scene_bundle['choices'])
        if [r['owner'] for r in select_targets(plan['inventory'])] != plan['selected_owner_ids']:
            raise ValueError('selection changed or post-qualification reselection occurred')
        if plan['GT_used'] or plan['legacy_panoptic_pixels_read'] or choices['future_candidate_scores_accessed']:
            raise ValueError('forbidden predictor access')
        frames = {int(f['frame_id']): f for f in plan['frames']}
        for owner, r in plan['owners'].items():
            if np.any(inputs.raw[inputs.g1.owner_ids == int(owner)] == 0):
                raise ValueError('raw-zero incumbent selected')
            with np.load(bundle_root/scene_bundle['supports'][owner], allow_pickle=False) as s:
                if len(s['xyz']) > 4096 or np.any(s['O'] & ~s['V']):
                    raise ValueError('physical-site bound or mask/visibility alignment differs')
                if not r['query_eligible']: continue
                bank = [r['anchor'], *[f for f in r['candidate_frame_ids'] if f != r['anchor']]]
                if len(bank) > 8 or len(set(bank)) != len(bank) or not all(r['views'][str(f)]['physical_qualified'] for f in bank):
                    raise ValueError('candidate bank qualification differs')
                pos = {int(f): i for i, f in enumerate(s['frame_ids'])}
                selected_rows = [pos[f] for f in bank]
                qd = choices['owners'][owner]['DISAGREEMENT']
                full_key = qd['visible_content_keys'][0]
                full = bundle['evidence'][full_key]['record']
                tiles = {bundle['evidence'][key]['record']['tile']: np.asarray(bundle['evidence'][key]['record']['scores'])
                         for key in qd['visible_content_keys'][1:] if bundle['evidence'][key]['record']['available']}
                interest, h = anchor_interest(s['area'], s['O'][pos[r['anchor']]], s['uv'][pos[r['anchor']]],
                    r['anchor_bbox'], np.asarray(full['scores']) if full['available'] else None,
                    tiles, predictor.valid_ids.index(r['old_class']), predictor.valid_ids.index(r['other_class']),
                    float(predictor.temperatures['F']))
                if h != qd['heterogeneity']: raise ValueError('anchor proxy failed exact reconstruction')
                for policy, chosen in choices['owners'][owner].items():
                    if chosen['candidate_semantic_scores_accessed'] or not chosen['choice_before_second_acquisition']:
                        raise ValueError('policy saw future candidate semantic scores')
                    if policy != 'DISAGREEMENT' and chosen['visible_evidence_keys']:
                        raise ValueError('simple policy saw anchor semantic evidence')
                    second, audit = score_bank(policy, xyz=s['xyz'], area=s['area'],
                        footprints=s['O'][selected_rows], center=r['center'],
                        cameras=[np.asarray(frames[f]['pose_c2w'])[:3, 3] for f in bank],
                        pixels=[r['views'][str(f)]['pixels'] for f in bank], frame_ids=bank,
                        interest=interest if policy == 'DISAGREEMENT' else None)
                    if second != chosen['second'] or any(audit[k] != chosen[k] for k in audit):
                        raise ValueError('sealed geometric policy did not reconstruct')
                    policies_count += 1
        for record in (r for r in registries if r['scene'] == scene):
            baseline = inputs.d2 if record['base'] == 'D2' else inputs.g1
            restored = restore_payload(record, baseline)
            item = verified(root/'predictions'/record['stage']/scene/'receipt.json')['methods'][record['method']]
            if restored.prediction_key != item['prediction_key']:
                raise ValueError('reconstructed whole payload differs from locked science')
            if record['base'] == 'G1':
                allowed = np.isin(inputs.g1.owner_ids, plan['selected_owner_ids'])
                if not np.array_equal(restored.semantic_labels[~allowed], inputs.g1.semantic_labels[~allowed]):
                    raise ValueError('unselected or recovered semantics changed')
            restored_count += 1
        for stage, content in scene_bundle['stages'].items():
            analysis = unpack_json(bundle_root/content['analysis'])
            for method, files in content['decisions'].items():
                decision = unpack_json(bundle_root/files['metadata'])
                payload = load_prediction(verified(root/'predictions'/stage/scene/'receipt.json')['methods'][method]['manifest'])
                labels = owner_labels(payload)
                if analysis['methods'][method]['class_agnostic'] != analysis['class_agnostic_fixed_support']:
                    raise ValueError('fixed-partition geometric negative control changed')
                with np.load(bundle_root/files['posteriors'], allow_pickle=False) as a:
                    if list(a['valid_ids']) != predictor.valid_ids or list(a['owner_ids']) != plan['selected_owner_ids']:
                        raise ValueError('portable posterior registry/vocabulary differs')
                    for j, owner in enumerate(a['owner_ids']):
                        r = plan['owners'][str(owner)]; d = decision['owners'][str(owner)]
                        with np.load(bundle_root/scene_bundle['supports'][str(owner)], allow_pickle=False) as s:
                            pos = {int(f): i for i, f in enumerate(s['frame_ids'])}; observations = []
                            for key in d['required_content_keys']:
                                e = bundle['evidence'][key]; meta = e['record']
                                with np.load(bundle_root/e['arrays'], allow_pickle=False) as feature:
                                    scores = feature['cosines'].astype(np.float64) if meta['available'] else None
                                    if meta['available'] and not np.array_equal(scores, meta['scores']):
                                        raise ValueError('portable cosine bytes differ from acquired values')
                                observations.append(Evidence(key, scores, s['O'][pos[meta['frame_id']]] if meta['available'] else None,
                                                             meta['available'], meta['reason']))
                            began_update = time.perf_counter()
                            result = update(a['p0'][j], observations, s['area'], d['updater'],
                                float(predictor.temperatures['F']), predictor.valid_ids, r['old_class'],
                                required_count=2 if r['query_eligible'] else 0)
                            aggregation_costs.setdefault(method, []).append(time.perf_counter()-began_update)
                            if not np.array_equal(a['p0'][j], r['p0']) or not np.array_equal(result['probability'], a['p_final'][j]):
                                raise ValueError('portable full-vocabulary posterior did not reconstruct exactly')
                            if result['label'] != labels[int(owner)] or result['weights'] != d['weights']:
                                raise ValueError('posterior output class/weights differ')
                            if result['p_new'] is not None and not np.array_equal(result['p_new'], a['p_new'][j]):
                                raise ValueError('portable new evidence distribution differs')
                            if method == 'DISAGREEMENT_MEAN' and int(owner) == engineering_objects.get(scene):
                                def recompute(rows):
                                    return update(a['p0'][j], rows, s['area'], 'SUPPORT',
                                        float(predictor.temperatures['F']), predictor.valid_ids, r['old_class'])
                                fixed = recompute(observations)
                                duplicated = recompute([observations[1], observations[0], observations[0]])
                                withdrawn = recompute(observations[:1])
                                withdrawn_reference = update(a['p0'][j], observations[:1], s['area'], 'AREA',
                                    float(predictor.temperatures['F']), predictor.valid_ids, r['old_class'])
                                empty = recompute([])
                                from scipy.special import softmax
                                single = softmax(observations[0].cosine/float(predictor.temperatures['F']))
                                if not np.array_equal(fixed['probability'], duplicated['probability']) or duplicated['duplicate_records'] != 1:
                                    raise ValueError('real fixed-set permutation/duplicate invariance failed')
                                if not np.array_equal(withdrawn['probability'], withdrawn_reference['probability']) or not np.allclose(withdrawn['p_new'], single, rtol=0, atol=1e-15) or not np.array_equal(empty['probability'], a['p0'][j]) or empty['label'] != r['old_class']:
                                    raise ValueError('real withdrawal/empty exact-prior invariant failed')
                                fixed_set_checks.append({'scene': scene, 'owner': int(owner), 'distinct_keys': fixed['unique_keys'],
                                    'duplicate_permutation_exact': True, 'withdrawal_recomputed_exact': True,
                                    'zero_evidence_prior_and_G1_class_exact': True, 'new_GPU_work': 0})
                            posterior_count += 1
    if (restored_count, posterior_count, policies_count) != (40, 512, 256):
        raise ValueError('restoration/causal-policy coverage differs')
    if len(fixed_set_checks) != 2: raise ValueError('real representative fixed-set check coverage differs')
    # Re-evaluate the exact gates from unrounded archived pools.
    pools = screen['pooled_metrics']
    qs, _ = choose_screen(pools, ['AREA_MEAN', 'COVERAGE_MEAN', 'VERIFY_MEAN'], screen['costs'])
    if qs.removesuffix('_MEAN') != selection['QS']: raise ValueError('simple query selection differs')
    us, _ = choose_screen(pools, [qs, 'QS_AREA'], screen['costs'])
    if ('MEAN' if us == qs else 'AREA') != selection['US']: raise ValueError('simple updater selection differs')
    for row in selection['comparisons']:
        metrics = lambda m: {c: pools[c][m]['metrics'] for c in pools}
        passed = screen_gate(metrics(row['candidate']), metrics(row['reference']), metrics('DQ01_G1'),
            changed=row['actual_prediction_changed'], corrections_net=row['corrections_net'], gt50_net=row['gt50_net'])
        if bool(passed) != row['passed_screen_extension_gate']: raise ValueError('gate result differs')
    if not selection['run_full_regression']:
        if regression['status'] != 'NOT_RUN_NO_SCREEN_SIGNAL' or timing['cold_calls'] != 0:
            raise ValueError('no-signal branch ran unrequested regression/cold acquisition')
        if list((root/'acquisition').glob('regression*/receipt.json')) or list((root/'cold').glob('calls/*/call.json')):
            raise ValueError('no-signal branch has unexpected additional inference')
    counts = verified(root/'costs.json')
    resources = binding['specification']['resources']; science = counts['science']; engineering = counts['engineering']
    if science.get('FC_encoding_attempts', 0) > resources['screen_unique_FC_images_max'] or science.get('region_pool_attempts', 0) > resources['screen_pool_head_max']:
        raise ValueError('science budget exceeded')
    if engineering.get('FULL_pool_attempts', 0)>4 or engineering.get('PROBE_pool_attempts', 0)>8 or engineering.get('FC_encoding_attempts', 0)>4:
        raise ValueError('engineering envelope exceeded')
    for row in screen['scene_metrics']:
        score = read(row['evaluation_receipt'])
        overlaps = np.asarray(row['runtime_overlaps'], float)
        validate_overlaps(overlaps)
        original = read(binding['baseline_rows']['SU01_G1'][row['scene']]['evaluation_receipt'])
        if score['identity'] != row['evaluation_identity'] or scorer_context(score['context']) != scorer_context(original['context']):
            raise ValueError('scorer context changed')
        confusion = np.asarray(score['confusion'], np.int64)
        if score['context']['minimum'] != 100 or confusion.shape != np.asarray(original['confusion']).shape or confusion.sum() != np.asarray(original['confusion'], np.int64).sum():
            raise ValueError('minimum or whole-scene semantic accounting differs')
    validation = verified(root/'execution/production_tests.json')
    if validation['command']['exit_code'] != 0 or '12 passed' not in Path(validation['command']['log']).read_text():
        raise ValueError('actual directed production test terminal proof missing')
    reference = read(root/'reference_validation.json')
    if reference['reference_exit_code'] != 0 or reference['actual_passed'] != 12:
        raise ValueError('supplied reference checks incomplete')
    for cohort, ordered in binding['screen_cohorts'].items():
        for method, pool in pools[cohort].items():
            expected = [next(r['evaluation_identity'] for r in screen['scene_metrics'] if r['scene']==s and r['method']==method) for s in ordered]
            if pool['ordered_inputs'] != expected or pool['ordered_scenes'] != ordered:
                raise ValueError('ordered pool input substitution')
    for path in reports['report_paths']:
        if not Path(path).is_file(): raise ValueError('required report missing')
    if verified(artifacts/'result_store.json')['identity'] != store['identity']:
        raise ValueError('Git artifact result store differs')
    index.write_memo(root/'verification/input_verifications.json')
    checks = [
        ('0', 'Fixed study and negative branch', '40 rows/20 pools; three gates recomputed; no extension or cold inference'),
        ('1', 'Repository, immutable actual parent and lineage', 'source_binding.json; SU/IR mapping; initial_state/storage_preflight'),
        ('2', 'New package, verbatim config, phase CLI', 'new modules and entrypoint; original spec; all phase dispatch'),
        ('3 / A-B', 'D-Surface and D-Vocabulary', 'four surface scenes; 26 vocabulary scenes; lossless original N/Q/F arrays and nested records'),
        ('4 / C-F', 'Targets, qualification and causal choices', 'actual predictor namespace; frozen inventory; 256 reconstructed policy choices; zero future-score access'),
        ('5 / G', 'Full vocabulary, distinct evidence and screen paths', '512 exact restored updates; 32 lossless posterior bundles; 40 whole payloads'),
        ('6', 'Conditional complete regression', 'NOT_RUN_NO_SCREEN_SIGNAL; fixed six-path conditional handler; no full target claim'),
        ('7 / H', 'Mechanism and costs', 'fixed G1 support; GT50/75 unique matching; tied-entry traces; logical vs union counts'),
        ('7 / I', 'Conditional cold timing', 'NOT_RUN_NO_FULL_TARGET_PASS; 0 calls; positive handler unexecuted in this task'),
        ('8', 'Proportionate validation and real engineering', '12 supplied/12 production tests; two actual FC parity objects; actual original scorer and ordered pool parity'),
        ('9', 'Freeze, resume and explicit statuses', 'unchanged operator SHA proofs; locked choices/predictions; completed leaf reuse; incompatible leaves rejected'),
        ('10', 'Three table families, four reports, portable evidence', 'CSV/LaTeX/Markdown; all 40 restored outputs/512 posteriors/256 choices; publication is a separate external receipt'),
        ('Evaluation 1', 'Stage accounting', '24/12 Stage A +16/8 Stage B; 10 methods; no independent scene inflation'),
        ('Evaluation 2-3', 'Ordered pooling and strict selection', 'original released thresholds and 100-point registry; exact four baseline pools; full target NOT_EVALUATED'),
        ('Evaluation 4', 'Fixed-support correction and geometric negative control', 'same support GT mapping; 8 undefined references retained; class-agnostic matches invariant'),
        ('Evaluation 5', 'Budget and independent policy cost', '35 scientific encodings/450 pool-heads; engineering 0/11; FULL budget matched, QD extra heads'),
        ('Evaluation 6-7', 'Timing boundary and completion', 'cached union stage timing only; no online FPS; true all CLI terminal proof recorded externally after return'),
    ]
    rows = [{'requirement': requirement, 'status': 'PASS_REALIZED_BRANCH', 'evidence': evidence, 'scope': name}
            for requirement, name, evidence in checks]
    rows.append({'requirement': 'Chronology disclosure', 'status': 'DISCLOSED_DEVIATION',
                 'scope': 'Original scorer parity completed after screen',
                 'evidence': 'Passed actual parity; no scientific operator, choice or gate was changed. Kept original engineering receipt as historical evidence.'})
    result = write(root/'verification/receipt.json', {'status': 'REALIZED_BRANCH_VERIFIED',
        'result_store_identity': store['identity'], 'reporting_identity': reports['identity'],
        'bundle_identity': bundle['identity'], 'scorer_parity_identity': parity['identity'],
        'restored_whole_payloads': restored_count, 'restored_owner_posteriors': posterior_count,
        'reconstructed_causal_policy_choices': policies_count, 'matrix': rows,
        'real_fixed_set_checks': fixed_set_checks,
        'tests_not_inflated_by_cells': True, 'publication_is_not_claimed_by_this_CLI': True})
    write(artifacts/'completion_matrix.json', result)
    import csv
    aggregation_path = artifacts/'tables/table3_CPU_aggregation_verification.csv'
    with aggregation_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'calls', 'mean_ms', 'total_ms', 'boundary'])
        writer.writeheader()
        for method, seconds in aggregation_costs.items():
            writer.writerow({'method': method, 'calls': len(seconds), 'mean_ms': 1000*float(np.mean(seconds)),
                'total_ms': 1000*sum(seconds), 'boundary': 'SINGLE_VERIFY_PASS_CPU_UPDATE_ONLY; NOT_COLD_OR_END_TO_END; NO_FILES_OR_FC'})
    markdown = '| Requirement | Status | Evidence |\n|---|---|---|\n'
    markdown += '\n'.join(f'| {r["requirement"]}: {r["scope"]} | {r["status"]} | {r["evidence"]} |' for r in rows)+'\n'
    (artifacts/'completion_matrix.md').write_text(markdown)
    write(root/'lifecycle_progress.json', {'status': 'REALIZED_BRANCH_CLI_COMPLETE_PUBLICATION_PENDING',
        'science_status': store['status'], 'verification_identity': result['identity'],
        'result_store_identity': store['identity'], 'goal_completed': False, 'published': False,
        'pending': ['actual_all_CLI_terminal_receipt', 'normal_push_and_remote_SHA_proof'],
        'new_science_FC_images': science.get('FC_encoding_attempts', 0),
        'new_science_pool_heads': science.get('region_pool_attempts', 0)})
    return result
