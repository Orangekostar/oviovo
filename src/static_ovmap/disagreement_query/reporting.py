"""Three table families, four evidence-bound reports and portable artifacts."""
import csv
import json
from pathlib import Path

import numpy as np

from .analysis import contrast
from .bundles import build_bundle
from .common import METRICS, REPO, canonical_digest, plain, read, verified, write


ARTIFACTS = REPO/'artifacts/static_ovmap/disagreement_query_v1'
REPORTS = REPO/'docs/paper/static_ovmap'


def table(folder, name, rows, columns=None):
    folder.mkdir(parents=True, exist_ok=True)
    columns = columns or list(rows[0])
    with (folder/(name+'.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    def value(item):
        if item is None: return '—'
        if isinstance(item, float): return f'{item:.4f}'
        return str(item)
    markdown = '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    markdown += '\n'.join('| '+' | '.join(value(row.get(k)).replace('|', '/') for k in columns)+' |' for row in rows)+'\n'
    (folder/(name+'.md')).write_text(markdown)
    def latex(item):
        return value(item).replace('_', r'\_').replace('%', r'\%').replace('—', '--')
    tex = '\\begin{tabular}{'+'l'*len(columns)+'}\n\\toprule\n'
    tex += ' & '.join(latex(k) for k in columns)+r' \\'+'\n\\midrule\n'
    tex += '\n'.join(' & '.join(latex(row.get(k)) for k in columns)+r' \\' for row in rows)
    tex += '\n\\bottomrule\n\\end{tabular}\n'
    (folder/(name+'.tex')).write_text(tex)
    return markdown


def report(binding):
    root = Path(binding['output_root']); ARTIFACTS.mkdir(parents=True, exist_ok=True)
    screen = verified(root/'screen_store.json'); selection = verified(root/'screen_selection.json')
    regression = verified(root/'regression.json'); timing = verified(root/'timing.json')
    parity = verified(root/'verification/original_scorer_parity.json')
    diagnostics = verified(root/'diagnostics/summary.json')
    scenes = [s for group in binding['screen_cohorts'].values() for s in group]
    pools = screen['pooled_metrics']; paths = screen['paths']
    method_stage = {m: 'screen_A' if m in verified(root/'evaluation/screen_A/store.json')['methods'] else 'screen_B'
                    for m in screen['methods']}
    plans = {s: verified(root/'plans'/s/'manifest.json') for s in scenes}
    choices = {s: verified(root/'choices'/s/'receipt.json') for s in scenes}
    analyses = {(stage, s): verified(root/'analysis'/stage/s/'receipt.json')
                for stage in ('screen_A', 'screen_B') for s in scenes}
    performance = []
    for method in screen['methods']:
        row = {'method': method}
        for cohort in binding['screen_cohorts']:
            for metric in METRICS:
                row[cohort+'_'+metric+'_percent'] = 100*pools[cohort][method]['metrics'][metric]
        performance.append(row)
    perf_md = table(ARTIFACTS/'tables', 'table1_performance_screen', performance)
    scene_rows = [{'scene': r['scene'], 'method': r['method'],
                   **{k: r['metrics'][k] for k in METRICS},
                   'prediction_identity': r['prediction_identity'], 'evaluation_identity': r['evaluation_identity'],
                   'reuse_kind': r['reuse_kind']} for r in screen['scene_metrics']]
    table(ARTIFACTS/'supplements', 'per_scene_metrics_fractions', scene_rows)
    pairs = [(selection['QD_US_screen_method'], selection['QS_US_screen_method']),
             ('QS_SUPPORT', selection['QS_US_screen_method']),
             ('QD_SUPPORT', selection['QD_US_screen_method']), ('QD_SUPPORT', 'QS_SUPPORT')]
    pairs += [(m, reference) for m in screen['methods'] if paths[m] is not None
              for reference in ('DQ01_G1', 'DQ00_D2')]
    contrasts = []
    for candidate, reference in pairs:
        # Class references are G1 supports. D2 has a different recovered partition,
        # so its accuracy delta is reported without a spurious fixed-G1 matching contrast.
        comparison = contrast(binding, candidate, reference, method_stage[candidate], method_stage[reference], scenes) if reference != 'DQ00_D2' else None
        row = {'candidate': candidate, 'reference': reference,
               'wrong_to_right': comparison['wrong_to_right'] if comparison else None,
               'right_to_wrong': comparison['right_to_wrong'] if comparison else None,
               'net_fixed_correct': comparison['corrections_net'] if comparison else None,
               'net_unique_GT50': comparison['gt50_net'] if comparison else None,
               'undefined_fixed_class': comparison['undefined_fixed_class_comparisons'] if comparison else None}
        row['net_unique_GT75'] = sum(analyses[(method_stage[candidate], s)]['methods'][candidate]['class_aware']['0.75']['count']
            -analyses[(method_stage[reference], s)]['methods'][reference]['class_aware']['0.75']['count'] for s in scenes) if comparison else None
        for cohort in pools:
            for metric in METRICS:
                row[cohort+'_'+metric+'_delta_pp'] = 100*(pools[cohort][candidate]['metrics'][metric]-pools[cohort][reference]['metrics'][metric])
        if paths[reference] is not None:
            pa, pb = paths[candidate][0], paths[reference][0]
            row['second_view_divergence'] = sum(choices[s]['owners'][o][pa].get('second') != choices[s]['owners'][o][pb].get('second')
                                              for s in scenes for o in plans[s]['owners'])
        else: row['second_view_divergence'] = None
        row['actual_prediction_changed'] = any(verified(root/'predictions'/method_stage[candidate]/s/'receipt.json')['methods'][candidate]['prediction_key']
            !=verified(root/'predictions'/method_stage[reference]/s/'receipt.json')['methods'][reference]['prediction_key'] for s in scenes)
        row['selected_owner_class_differences'] = sum(
            analyses[(method_stage[candidate], s)]['owner_rows'][o]['methods'][candidate]['class']
            !=analyses[(method_stage[reference], s)]['owner_rows'][o]['methods'][reference]['class']
            for s in scenes for o in plans[s]['owners']) if reference != 'DQ00_D2' else None
        row['whole_registry_rank_differences'] = sum(
            a_rank != dict(analyses[(method_stage[reference], s)]['methods'][reference]['current_ranks'])[owner]
            for s in scenes for owner, a_rank in analyses[(method_stage[candidate], s)]['methods'][candidate]['current_ranks']) if reference != 'DQ00_D2' else None
        gate = next((g['passed_screen_extension_gate'] for g in selection['comparisons']
                     if g['candidate'] == candidate and g['reference'] == reference), None)
        row['extension_gate_pass'] = gate
        contrasts.append(row)
    contrast_md = table(ARTIFACTS/'tables', 'table2_matched_contrasts', contrasts[:4],
        ['candidate', 'reference', 'replica_probe2_apall_delta_pp', 'cf_probe2_apall_delta_pp',
         'second_view_divergence', 'selected_owner_class_differences', 'wrong_to_right', 'right_to_wrong', 'net_unique_GT50', 'net_unique_GT75', 'extension_gate_pass'])
    table(ARTIFACTS/'supplements', 'all_contrasts_five_metrics', contrasts)
    acquisitions = {stage: verified(root/'acquisition'/stage/'receipt.json')
                    for stage in ('engineering_anchor', 'screen_anchor', 'screen_second')}
    science_counts = {}
    for stage in ('screen_anchor', 'screen_second'):
        for key, count in acquisitions[stage]['counts'].items(): science_counts[key] = science_counts.get(key, 0)+count
    failed = [verified(p) for p in (root/'acquisition').glob('*/failures/*.json')]
    cost_rows, ledger, degeneracies = [], [], {}
    for method in screen['methods']:
        cost = screen['costs'][method]; row = {'method': method, **cost}
        policy = paths[method][0] if paths[method] else None
        row.update(QD_fallback_count=0, sign_conflict_count=0, duplicate_records=0,
                   zero_support_fallback_count=0, support_equal_mean_weights_atol1e12=0, support_equal_area_weights_atol1e12=0,
                   support_exact_mean_pnew=0, support_exact_area_pnew=0, support_exact_mean_pfinal=0,
                   support_exact_area_pfinal=0, whole_scene_output_equal_mean=0, whole_scene_output_equal_area=0)
        if policy:
            stage = method_stage[method]
            for scene in scenes:
                decision = verified(root/'decisions'/stage/scene/(method+'.json'))
                mean_arrays = area_arrays = support_arrays = None
                if paths[method][1] == 'SUPPORT':
                    mean_method = selection['QS']+'_MEAN' if method == 'QS_SUPPORT' else 'DISAGREEMENT_MEAN'
                    area_method = 'QS_AREA' if method == 'QS_SUPPORT' else 'QD_AREA'
                    mean_arrays = np.load(root/'decisions/screen_A'/scene/(mean_method+'.npz'), allow_pickle=False)
                    area_arrays = np.load(root/'decisions/screen_B'/scene/(area_method+'.npz'), allow_pickle=False)
                    support_arrays = np.load(decision['lossless_posteriors']['path'], allow_pickle=False)
                    current_key = verified(root/'predictions'/stage/scene/'receipt.json')['methods'][method]['prediction_key']
                    row['whole_scene_output_equal_mean'] += current_key == verified(root/'predictions/screen_A'/scene/'receipt.json')['methods'][mean_method]['prediction_key']
                    row['whole_scene_output_equal_area'] += current_key == verified(root/'predictions/screen_B'/scene/'receipt.json')['methods'][area_method]['prediction_key']
                for owner, d in decision['owners'].items():
                    chosen = choices[scene]['owners'][owner][policy]; plan = plans[scene]['owners'][owner]
                    h = chosen.get('heterogeneity') or {}
                    row['QD_fallback_count'] += h.get('fallback') is not None
                    row['sign_conflict_count'] += bool(h.get('tile_sign_conflict'))
                    row['duplicate_records'] += d['duplicate_records']
                    row['zero_support_fallback_count'] += d['fallback'] is not None
                    with np.load(plan['support']['path'], allow_pickle=False) as a:
                        positions = {int(f): i for i, f in enumerate(a['frame_ids'])}
                        frames = [chosen.get('anchor'), chosen.get('second')]
                        if plan['query_eligible']:
                            footprint = a['O'][[positions[f] for f in frames]]
                            va = footprint.astype(float)@a['area']; shared = float(a['area'][footprint.all(axis=0)].sum())
                        else: va, shared = np.array([0., 0.]), 0.
                    w = np.asarray(d['weights'])
                    if paths[method][1] == 'SUPPORT' and len(w) == 2:
                        ordered_va = va[np.argsort(d['required_content_keys'])]
                        row['support_equal_mean_weights_atol1e12'] += bool(np.allclose(w/w.sum(), [.5, .5], rtol=0, atol=1e-12))
                        row['support_equal_area_weights_atol1e12'] += bool(va.sum()>0 and np.allclose(w/w.sum(), ordered_va/va.sum(), rtol=0, atol=1e-12))
                        j = d['p_final_array_row']
                        row['support_exact_mean_pnew'] += np.array_equal(support_arrays['p_new'][j], mean_arrays['p_new'][j])
                        row['support_exact_area_pnew'] += np.array_equal(support_arrays['p_new'][j], area_arrays['p_new'][j])
                        row['support_exact_mean_pfinal'] += np.array_equal(support_arrays['p_final'][j], mean_arrays['p_final'][j])
                        row['support_exact_area_pfinal'] += np.array_equal(support_arrays['p_final'][j], area_arrays['p_final'][j])
                    ar = analyses[(stage, scene)]['owner_rows'][owner]
                    baseline = analyses[('screen_A', scene)]['owner_rows'][owner]['methods']['DQ01_G1']
                    outcome = ar['methods'][method]
                    selected_geometry = chosen.get('candidates', {}).get(str(chosen.get('second')), {})
                    ledger.append({'scene': scene, 'owner': int(owner), 'method': method, 'policy': policy,
                        'updater': paths[method][1], 'selected': True, 'query_eligible': plan['query_eligible'],
                        'required_FULL_success': d['successful_required_FULL'], 'class_changed': d['changed'],
                        'c0': plan['old_class'], 'c1': plan['other_class'], 'final_class': outcome['class'],
                        'anchor': chosen.get('anchor'), 'second': chosen.get('second'),
                        'full_margin': h.get('full_margin'), 'tile_margins': json.dumps(h.get('tile_margins')),
                        'tile_priorities': json.dumps(h.get('tile_priorities')), 'tile_sign_conflict': h.get('tile_sign_conflict'),
                        'query_fallback': h.get('fallback'), 'J': selected_geometry.get('J'),
                        'H': selected_geometry.get('H'), 'theta_degrees': selected_geometry.get('theta_degrees'),
                        'coverage_novelty': selected_geometry.get('coverage_novelty'),
                        'anchor_O_area': float(va[0]), 'second_O_area': float(va[1]), 'overlap_O_area': shared,
                        'weights': json.dumps(d['weights']), 'duplicate_records': d['duplicate_records'],
                        'weight_content_keys': json.dumps(d['unique_keys']),
                        'required_content_keys': json.dumps(d['required_content_keys']),
                        'fixed_GT': ar['fixed_original_GT_id'], 'fixed_class': ar['fixed_class'],
                        'reference_undefined': ar['reference_undefined'], 'reference_ambiguous': ar['reference_ambiguous_class'],
                        'correct_before': baseline['correct_fixed_class'], 'correct_after': outcome['correct_fixed_class'],
                        'rank_before': baseline['current_rank'], 'rank_after': outcome['current_rank'],
                        'rank_changed': baseline['current_rank'] != outcome['current_rank'],
                        'second_differs_AREA': chosen.get('second') != choices[scene]['owners'][owner]['AREA'].get('second'),
                        'second_differs_QS': chosen.get('second') != choices[scene]['owners'][owner][selection['QS']].get('second')})
                if support_arrays is not None:
                    mean_arrays.close(); area_arrays.close(); support_arrays.close()
        cost_rows.append(row)
    cost_md = table(ARTIFACTS/'tables', 'table3_coverage_cost', cost_rows,
        ['method', 'selected', 'query_eligible', 'required_FULL_success', 'updated',
         'logical_FULL_reads', 'logical_probe_heads', 'QD_fallback_count', 'sign_conflict_count', 'duplicate_records'])
    degeneracy_md = table(ARTIFACTS/'tables', 'table3_support_degeneracy',
        [r for r in cost_rows if r['method'] in ('QS_SUPPORT', 'QD_SUPPORT')],
        ['method', 'support_equal_mean_weights_atol1e12', 'support_equal_area_weights_atol1e12',
         'support_exact_mean_pnew', 'support_exact_area_pnew', 'support_exact_mean_pfinal',
         'support_exact_area_pfinal', 'whole_scene_output_equal_mean', 'whole_scene_output_equal_area'])
    table(ARTIFACTS/'supplements', 'owner_evidence_correction_ledger', ledger)
    phase_costs = [{'phase': stage, 'category': r['category'], 'FC_encoding_attempts': r['counts'].get('FC_encoding_attempts', 0),
        'FULL_pool_attempts': r['counts'].get('FULL_pool_attempts', 0), 'PROBE_pool_attempts': r['counts'].get('PROBE_pool_attempts', 0),
        'pool_heads': r['counts'].get('region_pool_attempts', 0), 'worker_wall_seconds_including_load': r['elapsed_seconds'],
        'model_load_seconds': sum(m['seconds'] for m in r['model_loads']),
        'peak_allocated_GiB': r['peak_cuda_allocated_bytes']/2**30, 'peak_reserved_GiB': r['peak_cuda_reserved_bytes']/2**30,
        'standalone_policy_cold_timing': False} for stage, r in acquisitions.items()]
    phase_cost_md = table(ARTIFACTS/'tables', 'table3_actual_union_stages', phase_costs)
    surface, vocabulary = [], []
    for scene in scenes:
        receipt = verified(root/'diagnostics/surface'/scene/'receipt.json')
        for row in receipt['owners']:
            surface.append({'scene': scene, 'owner': row['owner'], 'query_eligible': row['query_eligible'],
                'legacy_rows_0': row['raw_source_rows']['0'], 'legacy_rows_1': row['raw_source_rows']['1'], 'legacy_rows_2plus': row['raw_source_rows']['2+'],
                'coordinate_0': row['co_located_legacy_coordinates']['0'], 'coordinate_1': row['co_located_legacy_coordinates']['1'],
                'coordinate_2plus': row['co_located_legacy_coordinates']['2+'],
                'legacy_coordinate_repeat_area_fraction': row['quadrature_legacy_coordinate_hits']['repeat_area_fraction'],
                'V_repeat_area_fraction': row['direct_geometric_V']['repeat_area_fraction'],
                'O_repeat_area_fraction': row['direct_FULL_supported_O']['repeat_area_fraction'],
                'V_outside_FULL_site_view_pairs': row['V_outside_FULL_site_view_pairs']})
    for group in binding['cohorts'].values():
        for scene in group:
            v = verified(root/'diagnostics/vocabulary'/scene/'receipt.json')
            vocabulary.append({k: v[k] for k in ('scene', 'incumbents', 'diagnosed_owners', 'record_count',
                'strict_probability_sign_changes_vs_pair', 'strict_probability_sign_changes_vs_FULL',
                'scaled_pair_sign_changes', 'FULL_D2_maximum_error', 'new_image_encodings', 'new_text_encodings')})
    table(ARTIFACTS/'supplements', 'surface', surface)
    table(ARTIFACTS/'supplements', 'vocabulary', vocabulary)
    eligibility = []
    for scene, plan in plans.items():
        for reason in sorted({r['reason'] for r in plan['inventory']}):
            rows = [r for r in plan['inventory'] if r['reason'] == reason]
            eligibility.append({'scene': scene, 'reason': reason, 'incumbents': len(rows),
                'selected': sum(r['owner'] in plan['selected_owner_ids'] for r in rows),
                'physically_query_eligible': sum(r['owner'] in plan['query_eligible_owner_ids'] for r in rows)})
    table(ARTIFACTS/'supplements', 'source_eligibility_coverage', eligibility)
    bundle = build_bundle(binding, screen)
    costs = write(root/'costs.json', {'science': science_counts,
        'engineering': acquisitions['engineering_anchor']['counts'], 'failed_attempts': failed,
        'actual_union_stages': phase_costs, 'policy_requirements': cost_rows,
        'direct_site_rays': sum(p['direct_site_rays'] for p in plans.values()),
        'geometry_preparation_scene_wall_seconds': {s: p['elapsed_seconds'] for s, p in plans.items()},
        'CPU_evaluation_scene_wall_seconds': {stage: {s: verified(root/'evaluation'/stage/s/'receipt.json')['elapsed_seconds'] for s in scenes}
            for stage in ('screen_A', 'screen_B')},
        'cold_timing_identity': timing['identity'], 'OS_page_cache_controlled': False,
        'NOT_end_to_end_or_online_FPS': True})
    store = write(root/'result_store.json', {'status': 'SCIENCE_COMPLETE_SCREEN_ONLY' if not selection['run_full_regression'] else 'SCIENCE_COMPLETE',
        **{k: screen[k] for k in ('cohorts', 'scene_metrics', 'pooled_metrics', 'methods', 'paths', 'costs',
            'scene_method_coverage', 'ordered_pool_coverage')},
        'binding_identity': binding['identity'], 'screen_store_identity': screen['identity'],
        'diagnostics_identity': diagnostics['identity'], 'selection': selection, 'regression': regression, 'timing': timing,
        'scorer_parity_identity': parity['identity'], 'portable_bundle_identity': bundle['identity'],
        'costs_identity': costs['identity'], 'deployment': 'N0_UNCHANGED', 'independent_confirmation': False,
        'metric_unit': 'FRACTION', 'implemented': 'REALIZED_BRANCH_AND_CONDITIONAL_HANDLERS',
        'executed': 'DIAGNOSTICS_26_AND_SCREEN_4', 'selected': regression['selected_method'],
        'publication_status': 'EXTERNAL_PUBLICATION_RECEIPT_AFTER_NORMAL_PUSH'})
    write(ARTIFACTS/'result_store.json', store)
    write(ARTIFACTS/'selection.json', selection); write(ARTIFACTS/'costs.json', costs)
    write(ARTIFACTS/'diagnostics.json', {'summary': diagnostics, 'surface': surface, 'vocabulary': vocabulary})
    write(ARTIFACTS/'contrasts.json', {'contrasts': contrasts})
    write(ARTIFACTS/'per_scene_metrics.json', {'rows': screen['scene_metrics']})
    write(ARTIFACTS/'regression_status.json', regression)
    (ARTIFACTS/'tables/table1_full_regression.md').write_text('NOT_RUN_NO_SCREEN_SIGNAL\n\n完整 8/18 场景回归未触发；无候选完整池数字。\n' if not selection['run_full_regression'] else '完整回归见外部 evaluation/regression/store.json。\n')
    signs = sum(v['strict_probability_sign_changes_vs_pair'] for v in vocabulary)
    records = sum(v['record_count'] for v in vocabulary)
    divergence = sum(choices[s]['owners'][o]['DISAGREEMENT'].get('second') != choices[s]['owners'][o]['VERIFY'].get('second')
                     for s in scenes for o in plans[s]['owners'])
    fallback = next(r['QD_fallback_count'] for r in cost_rows if r['method'] == 'DISAGREEMENT_MEAN')
    sign_conflicts = next(r['sign_conflict_count'] for r in cost_rows if r['method'] == 'DISAGREEMENT_MEAN')
    intro = ('固定筛选完成：10 条方法路径、4 个已曝光场景、40 条完整输出结果、20 个有序双场景池。'
             '三个预定扩展对比均失败，按协议保留 DQ01_G1，部署 N0_UNCHANGED。'
             '完整 8/18 场景回归 NOT_RUN_NO_SCREEN_SIGNAL；严格与材料目标未在完整回归中验证。\n\n')
    results = '# Disagreement query results\n\n'+intro
    results += '## 表 1：四场景开发筛选（%，不是 Replica8/CF18）\n\n'+perf_md
    results += '\n## 表 2：匹配机制对照（百分点）\n\n'+contrast_md
    results += '\n## 表 3：覆盖与成本\n\n'+cost_md+'\n'+phase_cost_md+'\n'+degeneracy_md
    results += '\n权重相等以归一化权重绝对容差 1e-12 判断；分布与整场输出相等为严格身份/逐元素相等。权重/分布分母为 64 对象，整场输出分母为 4 场景。\n'
    results += (f'\n科研联合采集新增 FC 图像编码 {science_counts.get("FC_encoding_attempts", 0)} 次、区域池化/head {science_counts.get("region_pool_attempts", 0)} 次；'
        '工程对象额外 4 FULL + 7 子区域 = 11 次池化/head，新增图像编码 0。筛选 B 完全复用已获取证据。'
        '采集阶段耗时包含缓存读取和模型加载，不能作为各独立策略的冷耗时或在线 FPS。条件冷测未触发。\n\n')
    results += (f'DISAGREEMENT 对 VERIFY 的第二视角选择分歧 {divergence}/64；回退 {fallback}/64；真实 tile 符号冲突 {sign_conflicts}/64。'
        f'词表诊断 {records} 条嵌套记录中有 {signs} 条概率差符号相对二类词表翻转，固定缩放分数差翻转 0；这些记录并非独立对象。'
        '几何分区、类无关匹配和恢复区域保持不变，无法据此宣称实例几何改善。\n\n')
    results += ('全部五项逐场景/逐类、GT50/75 唯一匹配、原始评分事件、逐对象权重、概率和诊断见 '
        '[结果包](../../../artifacts/static_ovmap/disagreement_query_v1/result_store.json) 与 [bundles](../../../artifacts/static_ovmap/disagreement_query_v1/bundles/manifest.json)。'
        '固定 G1 参考中有 8/64 类别映射未定义；纠错统计不把它们记成错误或从选择分母剔除。\n\n'
        '原始评分器一致性检查实际在筛选之后补齐；所有预测、查询和数值算子保持此前冻结版本，未根据该检查改变方法。'
        '两处真实 G1 评分和四个基线有序池一致性均通过。\n')
    (REPORTS/'DISAGREEMENT_QUERY_RESULTS.md').write_text(results)
    selection_text = '# Disagreement query selection\n\n'+intro
    selection_text += ('QS=COVERAGE；US=MEAN。选择使用未舍入 fraction 与固定字典序；未按数据集切换。\n\n'
        '扩展条件同时要求真实输出变化、两个池各自 APall 损失不超过 0.10 pp、平均 APall 增益至少 0.05 pp、'
        '相对 G1 的 APall/AP50/mIoU 保护，以及净固定参考纠错或净唯一 GT50 至少 +1。三个对比均未通过。\n\n')
    selection_text += contrast_md+'\nQS/US 的完整排序元组与门槛证据见 [selection.json](../../../artifacts/static_ovmap/disagreement_query_v1/selection.json)。\n'
    selection_text += '严格目标与材料目标：NOT_EVALUATED_FULL_REGRESSION_NOT_TRIGGERED。完整目标未验证，不将筛选数值当作完整 8/18 场景收益。\n'
    (REPORTS/'DISAGREEMENT_QUERY_SELECTION.md').write_text(selection_text)
    handoff = '# Disagreement query handoff\n\n'+intro
    handoff += ('代码、参数和原始任务包在本分支；父实验只读。绑定实际 234 行/18 池父结果，映射 SU00_D2/SU01_G1 与 IR 血缘。\n\n'
        '```bash\n/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_disagreement_query.py '
        '--spec configs/static_ovmap/disagreement_query_v1.json '
        '--parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 '
        '--output-root /mnt/shared/ww/ovimap-disagreement-query-v1/attempt_001 --phase all --resume\n```\n\n'
        '可单独使用 bind/diagnose/screen/expand/time/report/verify。expand 与 time 尊重已冻结条件，不接受手动强制扩展。'
        '新增算子与输入变更须先归档受影响的新任务后代；当前实现拒绝跨身份命中，原实验不会被修改。\n\n'
        '[紧凑结果](../../../artifacts/static_ovmap/disagreement_query_v1/result_store.json)、'
        '[可恢复清单](../../../artifacts/static_ovmap/disagreement_query_v1/bundles/manifest.json)、'
        '[完成矩阵](../../../artifacts/static_ovmap/disagreement_query_v1/completion_matrix.md)。'
        'bundle 包含小型 FULL/子区域 mask、特征/余弦、物理支持、损失无损概率、诊断、评分事件和完整 owner 类别/rank 注册表。'
        '原始 RGB-D、完整固定几何/owner 数组、模型权重与 dense 特征留在共享存储，内容身份和位置在清单及 source_binding 中。\n\n'
        '用 bundles.unpack_json 读取 gzip JSON；bundles.restore_payload(record, baseline) 由指定 G1/D2 owner 数组恢复完整输出，并核验 prediction_key/record_key。'
        'verify 阶段实际从 Git 工作树内 bundle 恢复全部 40 条输出和 512 条逐对象更新。\n\n'
        '物理代表样本最多 4096，属于确定性面积求积近似；不做容差焊接。FC 完整词表/FP32/池化/温度固定。'
        '科学采集实际使用 GPU 2、一处 FC 工作者、每工作者 4 CPU 线程；评估最多 3 进程。'
        '正分支回归与冷测处理器未在本次负筛选结果上执行，不能作为已验证性能。\n\n'
        f'权威外部目录：`{root}`。真实全 CLI 日志与退出码见 `execution/all_resume.commands.json`；'
        '正常 push 后完整本地/远端 SHA 与结果身份见外部 `publication/final.json`。后者不写入自引用提交。\n')
    (REPORTS/'DISAGREEMENT_QUERY_HANDOFF.md').write_text(handoff)
    claims = '# Disagreement query claims\n\n'+intro
    claims += ('| 结论 | 实测支持与边界 |\n|---|---|\n'
        '| 新选图优于最强简单规则 | 本轮不支持：QD+MEAN 相对 COVERAGE+MEAN 净纠错 −3、唯一 GT50 −3。 |\n'
        '| SUPPORT 聚合优于简单更新 | 本轮不支持：QS+SUPPORT 净纠错 −2；QD+SUPPORT 净纠错 0，扩展条件失败。 |\n'
        '| 概率分组有词表敏感性 | 原分数的固定嵌套词表存在符号翻转；不说明缩小词表会提升 AP，也不替换主融合规则。 |\n'
        '| 表面求积修复了实例几何 | 不成立：只用于测量/选图/聚合，实际 G1 owner 与几何不变。 |\n'
        '| 两次 FULL 读取算量一致 | FULL 读取预算匹配；QD 额外子区域 head，整体计算量不匹配。 |\n'
        '| 清理了历史 N/Q/F 混合误差 | 不成立：p0 原样保留，未拆解历史污染。 |\n'
        '| 获得已校准信息增益/GT 可靠性 | 不成立：异质性与覆盖均为固定代理，无该校准实验。 |\n'
        '| 完整回归/未见泛化/在线30 FPS收益 | 均未验证：四场景已曝光开发筛选，未触发完整回归及独立冷测。 |\n\n'
        '没有将缓存命中算作新推理；没有把原始 tied score entries 当作唯一 GT 匹配。'
        '固定参考歧义与最大匹配身份不唯一情况保留标记，净匹配基数用于门槛。'
        '实现、执行、选择、发布分别记录；外部发布凭据只有真实正常 push 并核对 SHA 后才生成。\n')
    (REPORTS/'DISAGREEMENT_QUERY_CLAIMS.md').write_text(claims)
    (ARTIFACTS/'README.md').write_text('# Portable disagreement-query results\n\n'+intro+
        'Three main table families are in tables/; diagnostics, owner ledgers and full metric contrasts are supplements/. '
        'bundles/manifest.json indexes lossless small inputs/results. Large immutable source owner/geometry arrays remain external.\n')
    return write(root/'reporting.json', {'status': 'REPORTS_AND_PORTABLE_BUNDLES_COMPLETE',
        'result_store_identity': store['identity'], 'bundle_identity': bundle['identity'],
        'report_paths': [str(REPORTS/('DISAGREEMENT_QUERY_'+name+'.md')) for name in ('RESULTS', 'HANDOFF', 'SELECTION', 'CLAIMS')],
        'three_main_table_families': ['performance', 'matched_contrasts', 'coverage_cost'],
        'input_identity': canonical_digest([screen['identity'], selection['identity'], regression['identity'], timing['identity'], parity['identity']])})
