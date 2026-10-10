"""Three evidence-only table families, compact learned weights and four reports."""
import csv
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .common import METRICS, REPO, ConsumptionIndex, verified, write
from .diagnostics import PAIRS, diagnose
from .evaluation import required_coverage
from .selection import select_study
from .resume import verify_completed_seed


def table1_rows(store, registry, parameter_counts):
    configs = {r['id']:r for r in registry}; rows = []
    for cohort in ('replica8','scannet_cf18'):
        for method in store['methods']:
            branch = method.removeprefix('R29_'); cfg = configs[branch]
            values = store['pooled_metrics'][cohort][method]['metrics']
            rows.append(dict(cohort=cohort,method=method,seed=29 if method.startswith('R29_') else 17 if cfg['trained'] else None,
                             trained=cfg['trained'],parameters=parameter_counts[branch] if cfg['trained'] else 0,
                             **{k:values[k] for k in METRICS}))
    return rows


def emit_table(folder, name, rows):
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    record = write(folder/(name+'.json'),dict(columns=columns,rows=rows,
                    units='Metrics in JSON/CSV are fractions; *_pp columns are percentage points; Markdown/LaTeX show percentage metrics'))
    def raw(value):
        return json.dumps(value,ensure_ascii=False,sort_keys=True) if isinstance(value,(dict,list)) else value
    with (folder/(name+'.csv')).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
        writer.writerows({k:raw(r.get(k)) for k in columns} for r in rows)
    def display(key,value):
        if value is None: return '—'
        if isinstance(value,bool): return 'yes' if value else 'no'
        if isinstance(value,float):
            return f'{100*value:.2f}' if key in (*METRICS,'top1','macro_recall') else f'{value:.5g}'
        return str(raw(value)).replace('|','\\|').replace('\n',' ')
    header = '| '+' | '.join(columns)+' |'
    lines = [header,'| '+' | '.join('---' for _ in columns)+' |']
    lines += ['| '+' | '.join(display(k,r.get(k)) for k in columns)+' |' for r in rows]
    (folder/(name+'.md')).write_text('\n'.join(lines)+'\n')
    def tex(value):
        return value.replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
    latex = [r'\begin{tabular}{'+'l'*len(columns)+'}',r'\toprule',' & '.join(tex(k) for k in columns)+r' \\',r'\midrule']
    latex += [' & '.join(tex(display(k,r.get(k))) for k in columns)+r' \\' for r in rows]
    latex += [r'\bottomrule',r'\end{tabular}']
    (folder/(name+'.tex')).write_text('\n'.join(latex)+'\n')
    return record


def table2_rows(root, dev, holdout, diagnostics, proposed):
    rows = []
    for a,b in PAIRS:
        first,second = dev['methods'][a],dev['methods'][b]
        keys_a = {r['key']:r for r in first['records']}; keys_b = {r['key']:r for r in second['records']}
        if keys_a.keys()!=keys_b.keys(): raise ValueError('DEV paired original-object denominator changed')
        accuracy = lambda rr:float(np.mean([p==r['class_id'] for r in rr['records'] for p in r['full200_top1']]))
        rows.append(dict(scope='DEV_BASE_SELECTED_HEAD',candidate=a,reference=b,original_objects=len(keys_a),
                         variant_records=4*len(keys_a),top1_delta_pp=100*(accuracy(first)-accuracy(second)),
                         equal_family_base_CE_delta=first['criterion']-second['criterion']))
    all_pairs = [*PAIRS,('R29_'+proposed,'R29_LR05_MA_8')]
    for a,b in all_pairs:
        for subset in ('base','heldout','all'):
            first,second = (holdout['metrics'][m][subset]['all_conditions'] for m in (a,b))
            if first['original_objects']!=second['original_objects']: raise ValueError('H paired denominator changed')
            rows.append(dict(scope='H_PROPOSAL_RECOGNITION',subset=subset,candidate=a,reference=b,
                original_objects=first['original_objects'],variant_records=first['variant_records'],
                top1_delta_pp=None if first['top1'] is None else 100*(first['top1']-second['top1']),
                macro_recall_delta_pp=None if first['macro_recall'] is None else 100*(first['macro_recall']-second['macro_recall']),
                NLL_delta=None if first['NLL'] is None else first['NLL']-second['NLL']))
    for contrast in diagnostics['comparisons']:
        if (contrast['candidate'],contrast['reference']) not in all_pairs: continue
        for threshold in ('0.5','0.75'):
            outcomes = contrast['semantic_outcomes'][threshold]
            rows.append(dict(scope='EXPOSED_WHOLE_MAP',cohort=contrast['cohort'],threshold=threshold,
                candidate=contrast['candidate'],reference=contrast['reference'],original_objects=contrast['original_incumbent_denominator'],
                F_wrong_to_right=outcomes['F'].get('WRONG_TO_RIGHT',0),F_right_to_wrong=outcomes['F'].get('RIGHT_TO_WRONG',0),
                final_wrong_to_right=outcomes['final'].get('WRONG_TO_RIGHT',0),final_right_to_wrong=outcomes['final'].get('RIGHT_TO_WRONG',0),
                undefined_GT_reference=outcomes['final'].get('UNDEFINED_GT_REFERENCE',0),
                gained_unique_GT=contrast['released_matches'][threshold]['counts'].get('new_unique_GT_matches',0),
                lost_unique_GT=contrast['released_matches'][threshold]['counts'].get('lost_unique_GT_matches',0),
                **{k+'_delta_pp':100*v for k,v in contrast['metric_deltas_fraction'].items()}))
    return rows


def coverage_rows(holdout, regression):
    rows = []
    for method,records in holdout['records'].items():
        prefix = {'LR02_FC_2':2,'LR03_FC_4':4}.get(method,8)
        histogram = {str(v):sum(r['available_views']==v for r in records) for v in range(2,prefix+1)}
        rows.append(dict(kind='ACTUAL_H_VIEW_PREFIX',method=method,requested_prefix=prefix,
                         original_objects=len({r['key'] for r in records}),variant_records=len(records),
                         actual_view_histogram=histogram,missing_requested_prefix=sum(r['available_views']<prefix for r in records),
                         repeated_padding_views=0))
    for scope,items in (('H',holdout['compact_summaries'].values()),('EXPOSED_MAP',regression['records'].values())):
        counts = {}
        for item in items:
            for method,value in item.get('models',item.get('summaries',{})).items():
                row = counts.setdefault(method,dict(scope=scope,method=method,logical_object_conditions=0,valid_local_tokens=0,
                    possible_local_tokens=0,MA_fallbacks=0,views_2=0,views_3_4=0,views_5_8=0))
                row['logical_object_conditions'] += 1
                row['valid_local_tokens'] += value['valid_local_tokens']; row['possible_local_tokens'] += value['possible_local_tokens']
                row['MA_fallbacks'] += value['fallback']=='NO_LOCAL_SUPPORT_MA_FALLBACK'
                v = value['available_views']; row['views_2' if v<=2 else 'views_3_4' if v<=4 else 'views_5_8'] += 1
        for row in counts.values():
            row['unavailable_local_fraction'] = 1-row['valid_local_tokens']/row['possible_local_tokens'] if row['possible_local_tokens'] else None
            rows.append(dict(kind='LOCAL_AND_VIEW_COVERAGE',**row))
    return rows


def proposal_coverage_rows(root):
    split = verified(root/'split_manifest.json'); rows = []
    for role,names in split['roles'].items():
        objects = [o for s in names for o in verified(root/'data/generated'/s/'manifest.json')['objects']]
        for prefix in (2,4,8):
            histogram = {str(v):sum(min(len(o['views']),prefix)==v for o in objects) for v in range(2,prefix+1)}
            rows.append(dict(kind='PROPOSAL_VIEW_PREFIX',scope=role,requested_prefix=prefix,original_objects=len(objects),
                             actual_view_histogram=histogram,missing_requested_prefix=sum(len(o['views'])<prefix for o in objects),
                             repeated_padding_views=0))
        for condition in ('truncate','append','truncate_append'):
            rows.append(dict(kind='FIXED_CORRUPTION_NO_OP',scope=role,condition=condition,original_objects=len(objects),
                             no_op_original_objects=sum(condition in o['no_op_conditions'] for o in objects)))
    return rows


def map_readout_stage(capture, regression):
    capture_seconds = capture.get('elapsed_seconds')
    seconds = None if capture_seconds is None else regression['elapsed_seconds']-capture_seconds
    if seconds is not None and (not math.isfinite(seconds) or seconds<0):
        raise ValueError('Regression capture/readout nested stage clocks are inconsistent')
    return dict(kind='NESTED_STAGE_COST',stage='MAP_READOUT_AFTER_CAPTURE',seconds=seconds,
                peak_allocated_GiB=regression['peak_allocated_bytes']/2**30,
                peak_reserved_GiB=regression['peak_reserved_bytes']/2**30,
                source_receipts=['recognition/regression.json','features/regression.json'],
                derivation=None if seconds is None else 'TOTAL_REGRESSION_SECONDS_MINUS_CAPTURE_SECONDS',
                timing_missing_reason='CAPTURE_CLOCK_NOT_RECORDED' if seconds is None else None,
                timing_boundary='CACHED_GRIDS_HEAD_LOAD_FORWARD_AND_COMPACT_ARRAY_WRITES',
                memory_boundary='READOUT_ONLY_AFTER_CAPTURE',conditional_on_cached_features=True,
                add_to_total=False)


def compute_rows(root, store, dev, holdout, regression):
    rows = []; specs = [('SUPERVISED_DATA_GENERATION','data/generation_runtime.json'),
                           ('H_DATA_GENERATION','data/holdout_generation_runtime.json'),
                           ('REGRESSION_BANK_GENERATION','regression/generation_runtime.json'),
                           ('FC_TRAIN_DEV_CAPTURE','features/train-dev.json'),('FC_H_CAPTURE','features/holdout.json'),
                           ('FC_MAP_CAPTURE','features/regression.json'),
                           ('DEV_SELECTED_INFERENCE','recognition/dev_selected.json'),('H_INFERENCE','recognition/holdout.json'),
                           ('MAP_CAPTURE_AND_INFERENCE','recognition/regression.json')]
    for stage,path in specs:
        source = verified(root/path)
        rows.append(dict(kind='NESTED_STAGE_COST' if stage=='FC_MAP_CAPTURE' else 'STAGE_COST',
                         stage=stage,seconds=source.get('elapsed_seconds',source.get('wall_seconds')),
                         peak_allocated_GiB=source.get('peak_allocated_bytes',0)/2**30 if 'peak_allocated_bytes' in source else None,
                         peak_reserved_GiB=source.get('peak_reserved_bytes',0)/2**30 if 'peak_reserved_bytes' in source else None,
                         source_receipt=path,memory_boundary='READOUT_ONLY_AFTER_CAPTURE' if stage=='MAP_CAPTURE_AND_INFERENCE' else None,
                         add_to_total=stage!='FC_MAP_CAPTURE',
                         memory_missing_reason='NOT_RECORDED_BEFORE_READOUT_MEMORY_RESET' if stage=='FC_MAP_CAPTURE' else None,
                         timing_missing_reason='CAPTURE_CLOCK_NOT_RECORDED' if stage=='FC_MAP_CAPTURE' and 'elapsed_seconds' not in source else None,
                         timing_boundary='LOCK_WAIT_MODEL_LOAD_AND_FEATURE_CAPTURE' if stage=='FC_MAP_CAPTURE' else
                                         'LOCK_WAIT_MODEL_LOAD_FEATURE_CAPTURE_AND_READOUT' if stage=='MAP_CAPTURE_AND_INFERENCE' else None))
    capture = verified(root/'features/regression.json')
    rows.append(map_readout_stage(capture,regression))
    profile_path = root/'runtime_profile/receipt.json'
    if profile_path.exists():
        profile = verified(profile_path)
        if profile['status']!='CACHE_ONLY_PROFILE_COMPLETE' or profile['original_readout']!=regression['identity']:
            raise ValueError('Conditional profile does not match the original readout')
        for stage,values in profile['stages'].items():
            rows.append(dict(kind='CONDITIONAL_CACHE_ONLY_PROFILE',stage=stage,**values,
                             conditional_on_cached_features=True,source_receipt='runtime_profile/receipt.json',
                             add_to_scientific_total=False,scientific_optimizer_updates=0,new_FC_image_encodings=0))
    for seed in (17,29):
        seed_row = verified(root/'training'/f'seed{seed}'/'receipt.json')
        for branch in ('WARMUP',*seed_row['branches']):
            row = verified(root/'training'/f'seed{seed}'/branch/'receipt.json')
            if row['status']!='COMPLETE' or row['completed_steps']!=2000: raise ValueError('Full fixed training budget absent')
            rows.append(dict(kind='STAGE_COST',stage='TRAINING_WITH_CHECKPOINT_DEV',seed=seed,method=branch,
                             optimizer_updates=row['completed_steps'],parameters=row['trainable_parameters'],seconds=row['elapsed_seconds'],
                             peak_allocated_GiB=row['peak_allocated_bytes']/2**30,peak_reserved_GiB=row['peak_reserved_bytes']/2**30,
                             source_receipt=f'training/seed{seed}/{branch}/receipt.json',
                             timing_boundary='ACCUMULATED_TRAINING_CHECKPOINT_AND_DEV_TIME_ACROSS_RESUMES',
                             memory_boundary='LAST_TRAINING_INVOCATION; EARLIER_RESUME_PEAKS_NOT_AGGREGATED',
                             auxiliary_counts=row['auxiliary_totals']))
    predict_phase = root/'phases/predict.json'
    invocation = verified(predict_phase) if predict_phase.exists() else None
    if invocation is not None and (invocation['status']!='COMPLETE' or invocation['phase']!='predict'):
        raise ValueError('Complete public prediction invocation required for its cost row')
    rows.append(dict(kind='CLI_INVOCATION_COST',stage='FULL_MAP_PREDICT_INVOCATION',
                     seconds=None if invocation is None else invocation['invocation_elapsed_seconds'],
                     source_receipt='phases/predict.json',
                     timing_missing_reason='PUBLIC_PREDICT_PHASE_NOT_RECORDED' if invocation is None else None,
                     timing_boundary='BANK_PREPARATION_ZERO_UPDATE_CHECK_FC_WORKER_AND_COMPLETE_CPU_PAYLOAD_RANK_CONSTRUCTION',
                     timing_scope='LAST_PUBLIC_PREDICT_INVOCATION; MAY_REUSE_EARLIER_CAPTURE_AND_READOUT_RECEIPTS',
                     add_to_total=False))
    rows.append(dict(kind='STAGE_COST',stage='EVALUATION_INVOCATION',seconds=store['elapsed_seconds'],CPU_workers=3))
    grids = [verified(p) for p in (root/'features/scientific/grids').glob('*.json')]
    rows.append(dict(kind='ACTUAL_SCIENCE_TOTAL',stage='DISTINCT_FC_INPUTS',distinct_FC_image_tensors=len(grids),
                     actual_FC_image_encodings=sum(r['physical_image_encodings'] for r in grids),optimizer_updates=16000,
                     image_budget_upper=3904,new_NQ_encodings=0,new_segmentation_calls=0,new_AnyUp_calls=0,new_SAMV_calls=0))
    if len(grids)>3904: raise ValueError('Fixed scientific FC image budget exceeded')
    engineer = verified(root/'engineering/receipt.json')
    rows.append(dict(kind='ENGINEERING_EXCLUDED_FROM_SCIENCE',stage='DISCARDED_FIT',optimizer_updates=engineer['optimizer_updates'],
                     seconds=engineer.get('elapsed_seconds'),selected_for_science=False))
    features = verified(root/'features/engineering.json')
    rows.append(dict(kind='ENGINEERING_EXCLUDED_FROM_SCIENCE',stage='ENGINEERING_FEATURE_CAPTURE',
                     actual_FC_image_encodings=features['physical_encodings_persisted'],seconds=features['elapsed_seconds'],
                     selected_for_science=False))
    for path in (root/'engineering').glob('failed_*.json'):
        failed = verified(path)
        rows.append(dict(kind='FAILED_ENGINEERING_CHECK',stage=path.stem,seconds=failed['elapsed_seconds'],
                         optimizer_updates=failed.get('fit_updates'),scientific_updates=failed['scientific_updates'],
                         failure_reason=failed['error']))
    for path in (root/'history').glob('*/regression/*/manifest.json'):
        previous = verified(path)
        rows.append(dict(kind='SUPERSEDED_GEOMETRY_BANK',stage='REGRESSION_BANK_REBUILD',scope=previous['scene'],
                         seconds=previous['elapsed_seconds'],GPU_inference=0,selected_for_science=False))
    parity = verified(root/'output_checks/summary.json')
    checked = [verified(root/'output_checks'/(scene+'.json')) for scene in parity['ordered_scenes']]
    rows.append(dict(kind='ENGINEERING_EXCLUDED_FROM_SCIENCE',stage='REAL_ZERO_UPDATE_WHOLE_OUTPUT_PARITY',
                     scenes=len(checked),seconds=sum(r['elapsed_seconds'] for r in checked),
                     GPU_inference=0,optimizer_updates=0,selected_for_science=False))
    return rows


def export_heads(binding, main, repeat, folder, index):
    import torch
    grouping = {r['id']:r['grouping'] for r in binding['specification']['methods']}
    records = {}
    for seed,nomination in ((17,main),(29,repeat)):
        for branch,source in nomination['checkpoints'].items():
            index.identity(source['path'],source)
            state = torch.load(source['path'],map_location='cpu',weights_only=False)
            if state['seed']!=seed or state['branch']!=branch: raise ValueError('Selected checkpoint identity differs')
            if any(v.dtype!=torch.float32 for v in state['model'].values()): raise ValueError('Released trainable head must be FP32')
            method = branch if seed==17 else 'R29_'+branch
            config = dict(grouping=grouping[branch],projected_channels=768,raw_channels=1536,
                          mask_in_chans=16,num_channels=256,num_output_maps=4,object_queries=4,local_hidden=128,
                          frozen_backbone_included=False,trained_category_parameters=0)
            destination = folder/'weights'/(method+'.pt'); destination.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model=state['model'],config=config,seed=seed,branch=branch,selected_step=state['step'],
                            training_checkpoint_sha256=source['sha256']),destination)
            artifact = index.identity(destination)
            if artifact['bytes']>=100*2**20: raise ValueError('Small head exceeds normal Git file limit')
            records[method] = dict(config=config,weights=artifact,selected_checkpoint=source,selected_step=state['step'])
    if len(records)!=6: raise ValueError('All four main and two repeat selected heads must be released')
    return write(folder/'learned_heads.json',dict(status='COMPLETE',heads=records))


def report(binding):
    root = Path(binding['output_root']); started = time.perf_counter()
    store = verified(root/'evaluation/store.json'); main = verified(root/'dev_nomination.json'); repeat = verified(root/'repeat_nomination.json')
    expected = [r['id'] for r in binding['specification']['methods']]+['R29_LR05_MA_8','R29_'+main['proposed_architecture']]
    if set(store['methods'])!=set(expected): raise ValueError('Full exact protocol method registry absent')
    coverage = required_coverage(binding['cohorts'],store['methods'],store['scene_metrics'],store['pooled_metrics'])
    if coverage!=(286,22): raise ValueError('Required 286 scene records and 22 complete ordered pools absent')
    seeds = [verify_completed_seed(binding,s) for s in (17,29)]
    if [r['scientific_updates'] for r in seeds]!=[10000,6000]: raise ValueError('Actual16000 scientific updates required')
    parity = verified(root/'output_checks/summary.json')
    if (parity['status']!='COMPLETE' or parity['ordered_scenes']!=[s for names in binding['cohorts'].values() for s in names]
            or len(parity['scene_receipts'])!=26 or not parity['whole_output_and_ranks_exact']):
        raise ValueError('Complete real26-scene zero-update parity evidence required')
    diagnostics = diagnose(binding,store); selection = select_study(binding,store)
    dev = verified(root/'recognition/dev_selected.json'); holdout = verified(root/'recognition/holdout.json')
    regression = verified(root/'recognition/regression.json'); profile = verified(root/'data_profile.json')
    h_profile = verified(root/'holdout_data_profile.json'); registry = binding['specification']['methods']
    folder = REPO/'artifacts/static_ovmap/learned_object_readout_v1'; folder.mkdir(parents=True,exist_ok=True)
    index = ConsumptionIndex(root/'publication/package_verifications.json')
    for source in ('split_manifest.json','class_split.json','source_binding.json','dependency_manifest.json','runtime_environment.json',
                   'data_profile.json','holdout_data_profile.json','engineering/receipt.json','features/text_mapping.json',
                   'features/regression.json',
                   'dev_nomination.json','repeat_nomination.json','selection.json','recognition/dev_selected.json','recognition/holdout.json',
                   'evaluation/store.json','diagnostics/summary.json','output_checks/summary.json'):
        src = root/source; index.identity(src); destination = folder/source; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,destination)
    profile_receipt = root/'runtime_profile/receipt.json'
    if profile_receipt.exists():
        index.identity(profile_receipt)
        (folder/'runtime_profile').mkdir(exist_ok=True)
        shutil.copyfile(profile_receipt,folder/'runtime_profile/receipt.json')
    predict_phase = root/'phases/predict.json'
    if predict_phase.exists():
        index.identity(predict_phase)
        (folder/'phases').mkdir(exist_ok=True)
        shutil.copyfile(predict_phase,folder/'phases/predict.json')
    for source in ('manifest.json','additional_mapping_reference.json'):
        src = root/'external/upstream_sources'/source; index.identity(src)
        dest = folder/'upstream'/source; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
    for scene,identity in parity['scene_receipts'].items():
        src = root/'output_checks'/(scene+'.json'); checked = verified(src)
        if checked['identity']!=identity or not all(checked[k] for k in
                ('geometry_exact','owner_array_exact','semantic_array_exact','official_current_ranks_exact','prediction_key_exact')):
            raise ValueError('Actual real-map zero-update receipt differs')
        index.identity(src); dest = folder/'output_checks'/src.name
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
    # Derived embeddings/scores and decisions only. Images, dense grids, GT maps and FC weights stay external.
    for scope,records in (('H',holdout['compact_summaries']),('regression',regression['records'])):
        for key,item in records.items():
            source = item.get('arrays') if scope=='H' else item['arrays']; index.identity(source['path'],source)
            dest = folder/'representations'/scope/Path(source['path']).name
            if scope=='regression': dest = dest.parent/item['scene']/dest.name
            else: dest = dest.parent/Path(source['path']).parent.name/dest.name
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source['path'],dest)
    for scene in binding['scenes']:
        for method in store['methods']:
            if method in ('LR00_D2','LR01_G1'): continue
            source = root/'decisions'/scene/(method+'.json'); index.identity(source)
            destination = folder/'decisions'/scene/(method+'.json'); destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
    heads = export_heads(binding,main,repeat,folder,index)
    split = verified(root/'split_manifest.json')
    train_objects = [o for s in split['roles']['train'] for o in verified(root/'data/generated'/s/'manifest.json')['objects'] if o['base']]
    actual_positive_ids = sorted({o['class_id'] for o in train_objects})
    write(folder/'supervision_scope.json',dict(loss_denominator_categories=160,adapter_heldout_categories=40,
          actual_positive_category_ids=actual_positive_ids,actual_positive_categories=len(actual_positive_ids),
          usable_original_base_objects=len(train_objects),training_families=24,
          class_sampling='UNIFORM_PRESENT_BASE_CLASS_THEN_UNIFORM_ORIGINAL_OBJECT'))
    whole_map_rows = table1_rows(store,registry,selection['parameter_counts'])
    table1 = emit_table(folder/'tables','table1_whole_map',[r for r in whole_map_rows if r['seed']!=29])
    table1_repeat = emit_table(folder/'tables','table1_repeat_seed29',[r for r in whole_map_rows if r['seed']==29])
    learning_rows = table2_rows(root,dev,holdout,diagnostics,main['proposed_architecture'])
    for row in learning_rows:
        for side in ('candidate','reference'):
            method = row[side]; branch = method.removeprefix('R29_')
            seed = 29 if method.startswith('R29_') else 17
            if branch in selection['parameter_counts']:
                trained = verified(root/'training'/f'seed{seed}'/branch/'receipt.json')
                warmup = verified(root/'training'/f'seed{seed}'/'WARMUP/receipt.json')
                row[side+'_trainable_parameters'] = trained['trainable_parameters']
                row[side+'_branch_updates'] = trained['completed_steps']
                row[side+'_warmup_updates'] = warmup['completed_steps']
            else:
                row[side+'_trainable_parameters'] = row[side+'_branch_updates'] = row[side+'_warmup_updates'] = 0
        row['paired_heads_share_warmup'] = row['candidate_warmup_updates']>0 and row['reference_warmup_updates']>0
    table2 = emit_table(folder/'tables','table2_learning',learning_rows)
    robust = []
    for method,subsets in holdout['metrics'].items():
        for subset,conditions in subsets.items():
            clean = conditions['clean']
            for condition in ('clean','truncate','append','truncate_append'):
                row = conditions[condition]
                robust.append(dict(kind='H_CONTROLLED_CORRUPTION',scope='PROPOSAL_RECOGNITION',method=method,subset=subset,
                    condition=condition,**row,top1_drop_from_clean_pp=None if row['top1'] is None else 100*(clean['top1']-row['top1']),
                    NLL_change_from_clean=None if row['NLL'] is None else row['NLL']-clean['NLL']))
    costs = compute_rows(root,store,dev,holdout,regression)
    table3 = emit_table(folder/'tables','table3_robustness_compute',
                        [*robust,*coverage_rows(holdout,regression),*proposal_coverage_rows(root),*costs])
    params = selection['parameter_counts']; nominated = main['proposed_architecture']
    h_fc = holdout['metrics']['LR04_FC_8']['all']['all_conditions']
    h_ma = holdout['metrics']['LR05_MA_8']['all']['all_conditions']
    h_proposed = holdout['metrics'][nominated]['all']['all_conditions']
    mechanism = [r for r in diagnostics['comparisons'] if (r['candidate'],r['reference']) in PAIRS]
    claims = write(folder/'claims.json',dict(whole_map_upgrade=selection['target_met'],material_upgrade=selection['material_target_met'],
                    exposed_benchmark_selection=True,independent_whole_map_confirmation=False,online_FPS_claim=False,
                    geometry_improvement_claim=False,novel_support=holdout['novel_support'],
                    nominated_method=nominated,paired_mechanism_evidence=mechanism,
                    DEV_proposed_minus_MA_CE=dev['methods'][nominated]['criterion']-dev['methods']['LR05_MA_8']['criterion'],
                    H_proposed_minus_MA_top1=holdout['metrics'][nominated]['all']['all_conditions']['top1']-holdout['metrics']['LR05_MA_8']['all']['all_conditions']['top1'],
                    H_MA_minus_FC8_top1=h_ma['top1']-h_fc['top1'],
                    H_proposed_minus_FC8_top1=h_proposed['top1']-h_fc['top1'],
                    repeat_consistency=selection['repeat_consistency'],no_automatic_mechanism_claim_from_gate=True,
                    auxiliary_supervision_actual_counts=verified(root/'training/seed17/LR08_MV_AUX/receipt.json')['auxiliary_totals']))
    write(folder/'method_registry.json',dict(main=registry,repeat=[m for m in expected if m.startswith('R29_')],parameters=params))
    # One run registry links real completed stages rather than treating test counts as science.
    write(root/'run_registry.json',dict(status='SCIENCE_COMPLETE',source_binding=binding['identity'],methods=expected,
          training={str(s):r['identity'] for s,r in zip((17,29),seeds)},data=profile['identity'],holdout_data=h_profile['identity'],
          DEV=dev['identity'],H=holdout['identity'],regression_readout=regression['identity'],evaluation=store['identity'],
          diagnostics=diagnostics['identity'],selection=selection['identity'],tables=[table1['identity'],table2['identity'],table3['identity']],
          table_supplements=dict(table1_repeat_seed29=table1_repeat['identity'])))
    shutil.copyfile(root/'run_registry.json',folder/'run_registry.json')
    result = write(root/'result_store.json',dict(**{k:v for k,v in store.items() if k not in ('identity','status','deployment')},
           status='SCIENCE_COMPLETE',science_identity=store['identity'],DEV_identity=dev['identity'],H_identity=holdout['identity'],
           diagnostics_identity=diagnostics['identity'],selection_identity=selection['identity'],scientific_optimizer_updates=16000,
           data_profile_identity=profile['identity'],selected_heads_identity=heads['identity'],claims_identity=claims['identity'],
           table_identities=[table1['identity'],table2['identity'],table3['identity']],
           table_supplements=dict(table1_repeat_seed29=table1_repeat['identity']),deployment='N0_UNCHANGED'))
    shutil.copyfile(root/'result_store.json',folder/'result_store.json')
    command = ('OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 '+binding['specification']['controller_python_hint']+
               ' -u scripts/evaluation/run_ovimap_learned_object_readout.py --spec configs/static_ovmap/learned_object_readout_v1.json'+
               ' --parent-root '+binding['parent_root']+' --observation-root '+binding['observation_root']+
               ' --output-root '+binding['logical_root']+' --phase all --resume')
    docs = REPO/'docs/paper/static_ovmap'; base = '../../../artifacts/static_ovmap/learned_object_readout_v1/'
    training_scope = (f"TRAIN:24 families, {profile['roles']['train']['base_objects']} original base objects; "
                      f"{profile['roles']['train']['base_categories']} observed positive categories within160 base categories. "
                      "40 adapter-heldout categories remain in the full200-class prediction vocabulary.")
    (docs/'LEARNED_OBJECT_RESULTS.md').write_text('# Learned object readout results\n\n'+training_scope+'\n\n'+
        f"DEV nomination: `{nominated}`. Research retained: `{selection['research_retained']}`; deployment: `N0_UNCHANGED`.\n\n"+
        f"Actual seed17/seed29 updates:10000/6000. Whole-map coverage:286 scene-method records,22 ordered pools.\n\n"+
        f"[Table1: whole-map regression]({base}tables/table1_whole_map.md) · [Table2: paired learning]({base}tables/table2_learning.md) · "+
        f"[Table3: robustness and compute]({base}tables/table3_robustness_compute.md)\n\n"+
        f"[Table1 supplement: seed29]({base}tables/table1_repeat_seed29.md). Parameter counts refer to trainable heads; the common frozen FC backbone is excluded.\n\n"+
        f"H full200-class top1 over all four conditions: FC8 {100*h_fc['top1']:.2f}%, MA8 {100*h_ma['top1']:.2f}%, "+
        f"DEV-nominated {nominated} {100*h_proposed['top1']:.2f}%. NLL: FC8 {h_fc['NLL']:.4f}, nominated {h_proposed['NLL']:.4f}. "+
        f"Denominator: {holdout['original_objects']} original objects and {h_fc['variant_records']} correlated condition records per method.\n\n"+
        'H measures GT-derived proposal recognition on new families; Replica8/CF18 are repeatedly exposed whole-map regressions. '+
        'Seed29 remains separate. All trained main branches are retained, including negative results. '+
        'Geometry and class-agnostic matches are unchanged; labels and current-class official area ranks can change.\n')
    (docs/'LEARNED_OBJECT_SELECTION.md').write_text('# Learned object selection\n\n'+json.dumps(selection,indent=2)+'\n\n'+
        'DEV checkpoint/architecture selection precedes H. The regression winner is exploratory on exposed benchmarks. '+
        'A main learned winner outside the pre-nominated repeat pair is a SINGLE_SEED_CANDIDATE. Deployment remains N0_UNCHANGED.\n')
    (docs/'LEARNED_OBJECT_CLAIMS.md').write_text('# Evidence boundaries\n\n'+
        f"Strict gate: {selection['target_met']}; material gate: {selection['material_target_met']}.\n\n"+
        'Interpret the matched MA–FC8, VIEW–MA, SURFACE–VIEW and AUX–SURFACE comparisons together with H and the fixed second-seed pair. '+
        'Passing the whole-map gate alone does not establish the novel mechanism. The complete contrast data are in Table2 and claims.json.\n\n'+
        f"H novel-category support: {holdout['novel_support']['status']}, {holdout['novel_support']['original_objects']} original objects.\n\n"+
        'The40 adapter-heldout categories supply no positive classification targets in this readout training. '+
        'Objects from these categories may still occur in image context or fixed corruptions; geometric membership supervision is separate. '+
        'Their absence from foundation-model pretraining is not established; scene-level backbone pretraining coverage cannot generally be audited.\n\n'+
        'New TRAIN/DEV/H sensor RGB is registered to the depth grid using measured calibration. Regression retains the inherited parent images and camera convention; '+
        'the two input pipelines are preserved separately when interpreting transfer from supervised proposals to predicted maps.\n\n'+
        'No online30FPS, cold end-to-end speed advantage, reconstructed-geometry improvement, untouched benchmark confirmation or global-SOTA claim. '+
        'Cached-feature timings are conditional model-resident readout costs. Checkpoint DEV validation is included in training; selected-head DEV inference is separately measured. '+
        'The original whole-map stage records a combined capture/readout time; its separate capture/readout clocks and capture VRAM were not recorded and remain null. '+
        'Readout peaks cover the post-capture interval. Any separately labeled cache-only profile is conditional on existing features and excluded from scientific totals. '+
        'Training wall time accumulates resumed invocations; memory peaks are from the final invocation and earlier resumed peaks were not aggregated. '+
        'The full public predict invocation includes CPU payload construction and current-class rank rebuilding. It may reuse earlier capture/readout receipts; '+
        'its invocation time and those earlier stage times do not form one cold end-to-end total.\n')
    (docs/'LEARNED_OBJECT_HANDOFF.md').write_text('# Reproduction and handoff\n\n'+training_scope+'\n\n'+
        'Run from this worktree:\n\n```bash\n'+command+'\n```\n\n'+
        f"FC worker environment: `{binding['FC_python']}`; GPU:{binding['gpu']}; inherited GPU lock is used.\n\n"+
        f"Source binding and all resolved paths: [source_binding.json]({base}source_binding.json). External raw/features/checkpoints: `{root}`.\n\n"+
        'Small selected FP32 head weights and configs: learned_heads.json and weights/. Load `ReadoutHead(checkpoint[\'config\'][\'grouping\'])`, '+
        'then `head.load_state_dict(checkpoint[\'model\'], strict=True)`; use the exact bound frozen FC phi and original frozen text. '+
        'The weights contain no FC backbone or text encoder.\n\n'+
        'Each of the three table families has Markdown/CSV/JSON/LaTeX. Their numerical sources are the compact result store, '+
        'H recognition, DEV selected-head receipts, per-object decisions and post-lock diagnostics.\n\n'+
        'runtime_profile/receipt.json records one cache-only replay with separate capture/readout clocks, zero new encodings and zero updates. '+
        'It is reproduced by the predict worker on a fresh output root and reused on resume; it is not cold end-to-end latency. '+
        'publication/package.json records packaging wall time separately.\n\n'+
        'The full CLI terminal exit and local/remote publication SHA are verified externally after the process exits; '+
        'see publication/final.json in the external output root. The release commit never embeds its own future SHA.\n')
    (folder/'README.md').write_text('# Learned object readout evidence\n\n'+training_scope+'\n\n'+
        'See docs/paper/static_ovmap/LEARNED_OBJECT_HANDOFF.md for the exact full reproduction command. '+
        'Tables are generated from actual complete results; raw scans, GT maps, dense grids and frozen weights remain external. '+
        'Selected trainable heads retain the MaskAdapter license in the source module.\n')
    write(root/'publication/package.json',dict(status='PACKAGED',result_store_identity=result['identity'],
           selected_heads=heads['identity'],artifact_root=str(folder),elapsed_seconds=time.perf_counter()-started,
           raw_training_data_copied=False,frozen_weights_copied=False,publication_observed=False))
    (folder/'publication').mkdir(exist_ok=True)
    shutil.copyfile(root/'publication/package.json',folder/'publication/package.json')
    index.write_memo(root/'publication/package_verifications.json')
    return result
