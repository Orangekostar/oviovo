"""Three evidence-bound table families, portable heads and explicit completion paths."""
import gzip
import json
import shutil
from functools import cmp_to_key
from pathlib import Path
import time
import numpy as np
from .common import REPO,METRICS,ConsumptionIndex,read,verified,write
from static_ovmap.preservation_routing.tabulate import write_table

def optional(path):return verified(path) if Path(path).exists() else None

def pct(v):return None if v is None else 100*float(v)

def map_selection(binding,store,lock):
    result=dict(DEV_NOMINEE=lock.get('DEV_nominee'),EXPOSED_BENCHMARK_CANDIDATE=None,
        benchmark_selection_exploratory=True,research_retained='REF_G1',deployment='N0_UNCHANGED',gates={},latency_gate=False)
    if not lock.get('activated_full_path'):
        return dict(result,status='NOT_TRIGGERED_BY_DEV_REPEAT',same_architecture_repeat=None,repeat_map_pass=None)
    if not store or store.get('status')!='EVALUATION_COMPLETE':return dict(result,status='PARTIAL_ACTIVE_STAGE')
    pools=store['pooled_metrics'];gates={};passing=[]
    order=[m['id'] for m in binding['specification']['methods']]
    for name in order+['S29_'+m for m in order]:
        checks={c:{k:pools[c][name]['metrics'][k]>=pools[c]['REF_G1']['metrics'][k]-1e-10 for k in METRICS} for c in binding['cohorts']}
        cf=pools['scannet_cf18'][name]['metrics'];d2=pools['scannet_cf18']['REF_D2']['metrics']
        checks['CF_vs_D2']=dict(APall_gt=cf['apall']>d2['apall']+1e-10,AP50_ge=cf['ap50']>=d2['ap50']-1e-10)
        passed=all(v for row in checks.values() for v in row.values());gain=cf['apall']-d2['apall']
        gates[name]=dict(passed=passed,checks=checks,CF_APall_gain_fraction=gain,material_CF_gain=gain>=.001-1e-10)
        if passed and not name.startswith('S29_'):passing.append(name)
    def compare(a,b):
        aa=[pools['scannet_cf18'][a]['metrics']['apall'],pools['replica8'][a]['metrics']['apall'],sum(pools[c][a]['metrics']['miou'] for c in pools)]
        bb=[pools['scannet_cf18'][b]['metrics']['apall'],pools['replica8'][b]['metrics']['apall'],sum(pools[c][b]['metrics']['miou'] for c in pools)]
        for x,y in zip(aa,bb):
            if abs(x-y)>1e-10:return -1 if x>y else 1
        return order.index(a)-order.index(b)
    winner=sorted(passing,key=cmp_to_key(compare))[0] if passing else None
    result.update(status='TARGET_MET_RESEARCH_ONLY' if winner else 'COMPLETE_NO_TARGET_GAIN',gates=gates,
        EXPOSED_BENCHMARK_CANDIDATE=winner,research_retained=winner or 'REF_G1',same_architecture_repeat='S29_'+winner if winner else None,
        repeat_map_pass=gates['S29_'+winner]['passed'] if winner else None)
    return result

def completion(binding,lock,evaluation):
    root=Path(binding['output_root']);missing=[]
    for name in ('prepare.json','prior_audit/receipt.json','engineering/receipt.json','engineering/two_scene.json',
                 'engineering/sparse_loader_parity.json','engineering/bounded_validation.json','diagnostic_D1/receipt.json','predictor_lock.json',
                 'diagnostics/receipt.json','export/weights_registry.json','export/roundtrip.json'):
        row=optional(root/name)
        if row is None or row.get('status') not in ('COMPLETE','PREDICTORS_LOCKED'):missing.append(name)
    updates=0;arms=[]
    for p in sorted((root/'training').glob('seed*/*/receipt.json')):
        r=verified(p);updates+=r.get('completed_steps',0)
        if r['status']=='COMPLETE':arms.append(r)
    expected=lock.get('scientific_updates_expected')
    if expected is None or updates!=expected or len(arms)!=expected//2000:missing.append('activated_training_coverage')
    if lock.get('activated_full_path'):
        if not evaluation or (evaluation.get('scene_method_coverage'),evaluation.get('full_cohort_pool_coverage'))!=(286,22):missing.append('full_map_286_22')
        for name in ('holdout2/receipt.json','real_proposals/receipt.json','confirmation/receipt.json'):
            row=optional(root/name)
            if row is None or row.get('status') not in ('COMPLETE','LIMITED_CONFIRMATION_SUPPORT'):missing.append(name)
    return missing,updates,arms

def table_families(binding,folder,lock,evaluation,arms):
    root=Path(binding['output_root']);tables={};out=folder/'tables'
    def table(name,rows,columns=None):
        if columns is None:
            columns=list(dict.fromkeys(k for r in rows for k in r))
            rows=[{k:r.get(k) for k in columns} for r in rows]
        tables[name]=write_table(out,name,columns,rows)
    # Family1: inherited references stay explicit when the scientific map path is inactive.
    parent=verified(Path(binding['lr_parent_root'])/'result_store.json');map_rows=[]
    for cohort in binding['cohorts']:
        for name in binding['specification']['whole_map']['main_methods']+binding['specification']['whole_map']['repeat_methods']:
            pool=(evaluation or {}).get('pooled_metrics',{}).get(cohort,{}).get(name)
            if pool:metrics=pool['metrics'];status='MEASURED_OFFICIAL_ORDERED_POOL'
            elif name in binding['specification']['map_baselines']:
                metrics=parent['pooled_metrics'][cohort][binding['specification']['map_baselines'][name]]['metrics'];status='INHERITED_LR_REFERENCE'
            else:metrics={k:None for k in METRICS};status='NOT_TRIGGERED_BY_DEV_REPEAT' if not lock.get('activated_full_path') else 'MISSING_ACTIVE_STAGE'
            map_rows.append(dict(cohort=cohort,method=name,seed=29 if name.startswith('S29_') else 17 if name.startswith('FG') and name!='FG0_FC8' else None,
                status=status,**dict(zip(('APall_pct','AP50_pct','AP25_pct','mIoU_pct','mAcc_pct'),[pct(metrics[k]) for k in METRICS]))))
    table('table1_whole_map',[r for r in map_rows if r['seed']!=29]);table('table1_whole_map_repeat',[r for r in map_rows if r['seed']==29])
    # Family2: DEV selection, exact-step contrasts and descriptive diagnostics.
    dev=[];reference=optional(root/'recognition/dev_FC8.json')
    if reference:dev.append(dict(method='FC8',seed=None,step=None,status='FROZEN_REFERENCE',**reference['metrics']))
    for r in arms:dev.append(dict(r['selected_metrics'],method=r['branch'],seed=r['seed'],status='QUALIFIED' if r['qualified'] else 'BEST_DIAGNOSTIC_NOT_QUALIFIED'))
    columns=['method','seed','step','status','A_pct','M_pct','C_pct','CE_base160','NLL_full200','corrections','harm','net_corrections','original_objects','variant_records']
    displayed=[dict(method=r['method'],seed=r['seed'],step=r['step'],status=r['status'],A_pct=pct(r['A']),M_pct=pct(r['M']),C_pct=pct(r['C']),
        CE_base160=r['CE'],NLL_full200=r['NLL'],**{k:r[k] for k in columns[9:]}) for r in dev]
    table('table2_DEV_selection',displayed,columns)
    contrasts=[];by={(r['seed'],r['branch']):r for r in arms}
    pairs=[('FG2_SURFACE','FG1_VIEW'),('FG3_AUX','FG2_SURFACE'),('FG4_ROUTE','FG3_AUX')]
    for seed in (17,29):
        for a,b in pairs:
            if (seed,a) not in by or (seed,b) not in by:continue
            first,second=by[(seed,a)],by[(seed,b)]
            values=[('INDEPENDENT_SELECTED_STEPS',first['selected_metrics'],second['selected_metrics'])]
            values += [('SAME_FIXED_STEP',next(r for r in first['validations'] if r['step']==step),next(r for r in second['validations'] if r['step']==step)) for step in (250,500,1000,1500,2000)]
            for scope,aa,bb in values:contrasts.append(dict(seed=seed,contrast=a+' minus '+b,scope=scope,first_step=aa['step'],second_step=bb['step'],
                delta_A_pp=100*(aa['A']-bb['A']),delta_M_pp=100*(aa['M']-bb['M']),delta_C_pp=100*(aa['C']-bb['C']),
                delta_CE=aa['CE']-bb['CE'],original_objects=69,variant_records=276))
    table('table2_matched_contrasts',contrasts,['seed','contrast','scope','first_step','second_step','delta_A_pp','delta_M_pp','delta_C_pp','delta_CE','original_objects','variant_records'])
    for name,path in [('recognition','diagnostics/recognition_summary.json'),('H2','holdout2/recognition_summary.json')]:
        receipt=optional(root/path)
        if receipt:
            rows=[]
            for row in receipt['rows']:
                r=dict(row);r['top1_pct']=pct(r.pop('top1'));r['macro_pct']=pct(r.pop('macro'));rows.append(r)
            table('table2_'+name,rows)
    d0=optional(root/'prior_audit/receipt.json')
    if d0:
        rows=[]
        for r in d0['summary']:
            rows.append(dict(role=r['role'],method=r['method'],scope=r['scope'],original_objects=r['original_objects'],variant_records=r['variant_records'],
                top1_pct=pct(r['top1']),corrections=r['corrections'],harm=r['harm'],GT_margin_mean=r['GT_margin']['mean'],
                baseline_GT_margin_mean=r['baseline_GT_margin']['mean'],baseline_top1_top2_gap_mean=r['baseline_winner_runner_gap']['mean'],
                L2_drift_mean=r['L2_drift']['mean'],GT_DESCRIPTIVE_ONLY=True))
        table('table2_D0',rows)
    d1=optional(root/'diagnostic_D1/receipt.json')
    if d1:
        rows=[]
        for scope,values in [('ALL_ORIGINALS',d1['summary']),('COMMON_NONFALLBACK',d1['paired_nonfallback'])]:
            for value in values:
                r={k:v for k,v in value.items() if k!='paired_scope'};r['pair_scope']=scope
                r['top1_pct']=pct(r.pop('top1'));r['macro_pct']=pct(r.pop('macro'));rows.append(r)
        table('table2_D1',rows)
        lookup={(r['role'],r['pair_scope'],r['scope'],r['method']):r for r in rows};delta=[]
        for (role,pair,scope,method),r in lookup.items():
            if method!='TARGET_ONLY':continue
            b=lookup[(role,pair,scope,'ALL')]
            delta.append(dict(role=role,pair_scope=pair,scope=scope,contrast='TARGET_ONLY minus ALL',
                delta_top1_pp=None if r['top1_pct'] is None else r['top1_pct']-b['top1_pct'],
                delta_macro_pp=None if r['macro_pct'] is None else r['macro_pct']-b['macro_pct'],
                delta_NLL=None if r['NLL'] is None else r['NLL']-b['NLL'],original_objects=r['original_objects'],
                variant_records=r['variant_records'],GT_DIAGNOSTIC_ONLY=True))
        table('table2_D1_contrasts',delta)
    intervention_rows=[];mass_rows=[];intervention_delta=[]
    for p in sorted((root/'diagnostics').glob('*/*/receipt.json'))+sorted((root/'holdout2/diagnostics').glob('*/receipt.json')):
        r=verified(p);mass_rows.append(dict(role=r['role'],method=r['method'],original_objects=r['original_objects'],variant_records=r['variant_records'],
            **r['mass_means'],membership_status=r['membership']['status'],membership_accuracy=r['membership'].get('accuracy'),
            membership_Brier=r['membership'].get('Brier'),membership_AUROC=r['membership'].get('AUROC'),membership_known=r['membership'].get('known'),
            contribution_norm_is_proxy=True))
        cells={}
        for v in r['summary']:
            scope=v.get('paired_scope','ORDINARY_FULL_SUPPORT');cells[(v['method'],v['scope'],scope)]=v
            rr=dict(v,role=r['role'],seed_method=r['method'],paired_scope=scope);rr['top1_pct']=pct(rr.pop('top1'));rr['macro_pct']=pct(rr.pop('macro'));intervention_rows.append(rr)
        for (method,scope,pair),v in cells.items():
            if method=='FG4_ROUTE':continue
            base=cells.get(('FG4_ROUTE',scope,pair))
            if base is None:continue
            intervention_delta.append(dict(role=r['role'],seed_method=r['method'],scope=scope,paired_scope=pair,
                contrast='FG4_ROUTE minus '+method,delta_top1_pp=None if v['top1'] is None else 100*(base['top1']-v['top1']),
                delta_macro_pp=None if v['macro'] is None else 100*(base['macro']-v['macro']),delta_NLL=None if v['NLL'] is None else base['NLL']-v['NLL'],
                original_objects=v['original_objects'],variant_records=v['variant_records'],diagnostic_only=True))
    if intervention_rows:table('table2_interventions',intervention_rows);table('table2_intervention_contrasts',intervention_delta)
    if mass_rows:table('table2_membership_and_mass',mass_rows)
    real=optional(root/'real_proposals/receipt.json')
    if real:
        table('table2_real_proposals',[dict(method=m,quality=q,proposals=v['proposals'],unique_GT=v['unique_GT'],
            proposal_top1_pct=pct(v['proposal_top1']),equal_GT_top1_pct=pct(v['equal_GT_top1']),purity_metrics='NOT_COMPUTED')
            for m,qualities in real['metrics'].items() for q,v in qualities.items()])
        real_diag=verified(root/'real_proposals/diagnostics.json');rr=[]
        for v in real_diag['summary']:
            row=dict(v);row['top1_pct']=pct(row.pop('top1'));row['macro_pct']=pct(row.pop('macro'));rr.append(row)
        table('table2_real_interventions',rr or [dict(status='LIMITED_CONFIRMATION_SUPPORT',variant_records=0)])
    # Family3: actual coverage and costs. Never sum nested/resumed clocks as a cold total.
    costs=[]
    for r in arms:
        costs.append(dict(scope='TRAIN_ARM',method=r['branch'],seed=r['seed'],status=r['status'],optimizer_updates=r['completed_steps'],
            original_draws=r['original_item_draws'],head_forward_calls=r['clean_corrupt_forward_calls'],
            architecture_parameters=r['architecture_parameters'],loss_participating_parameters=r['loss_participating_parameters'],
            elapsed_seconds=r['elapsed_seconds'],peak_allocated_GiB=r['peak_allocated_bytes']/2**30,peak_reserved_GiB=r['peak_reserved_bytes']/2**30,
            replayed_updates=r['discarded_uncheckpointed_updates'],timing='RESIDENT_CACHED_INPUTS_INCLUDING_DEV_CHECKPOINT_EVALUATION'))
    phase_files=sorted((root/'phases').glob('*.json'))+sorted((root/'phase_history').glob('*/*.json'))
    for p in phase_files:
        r=verified(p);costs.append(dict(scope='LAST_OBSERVED_PHASE' if p.parent.name=='phases' else 'PRIOR_OBSERVED_PHASE',
            method=r['phase'],attempt=r['attempt'],source_receipt=str(p.relative_to(root)),status=r['status'],elapsed_seconds=r['stage_elapsed_seconds'],
            timing='PHASE_INVOCATION_MAY_REUSE_COMPLETED_RECEIPTS_DO_NOT_SUM_AS_COLD',worker_cost=r.get('worker_cost')))
    table('table3_cost',costs or [dict(scope='NO_COMPLETED_COST',status='PARTIAL_ACTIVE_STAGE')])
    coverage=[]
    for p in sorted((root/'recognition').glob('*/*.json'))+sorted((root/'holdout2/recognition').glob('*.json')):
        value=verified(p)
        if 'records' not in value or not isinstance(value['records'],list):continue
        counts=[c for r in value['records'] for c in r.get('coverage',[])];frames={(r['family'],f) for r in value['records'] for c in r.get('coverage',[]) for f in c['frame_ids']}
        coverage.append(dict(scope=p.parent.name,method=p.stem,original_objects=value['original_objects'],variant_records=value['variant_records'],
             distinct_frames=len(frames),view_observations=sum(c['actual_views'] for c in counts),local_observations=sum(c['local_observations'] for c in counts),
             local_support_fallbacks=sum(c['fallback'] is not None for c in counts),physical_image_encodings=value['physical_image_encodings'],
             head_or_base_forward_calls=value['head_or_base_forward_calls']))
    for cohort,scenes in binding['cohorts'].items():
        for scene in scenes:
            manifest=verified(Path(binding['lr_parent_root'])/'regression'/scene/'manifest.json')
            counts={}
            for owner in manifest['inventory']:counts[owner['reason']]=counts.get(owner['reason'],0)+1
            coverage.append(dict(scope='FIXED_MAP_REGISTRY',cohort=cohort,scene=scene,status='INHERITED_GEOMETRY_ONLY',
                total_positive_owners=len(manifest['full_G1_positive_registry']),eligible_owners=len(manifest['eligible_owners']),**counts))
    from .common import objects
    train_objects=objects(binding,'train')
    for arm in arms:
        directory=root/f'training/seed{arm["seed"]}'/arm['branch']
        with np.load(directory.parent/'draws.npz',allow_pickle=False) as a:draws=a['draws'][:arm['completed_steps']*16].copy()
        seen=set();frames=set();view_reads=0
        for oi,prefix,condition in draws:
            obj=train_objects[int(oi)];seen.add(obj['key']);selected=obj['views'][:int(prefix)]
            frames.update((obj['scene'],int(v['frame_id'])) for v in selected);view_reads+=2*len(selected)
        logs=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
        totals={k:sum(row['denominators'][k] for row in logs) for k in ('positive_tokens','negative_tokens','unknown_tokens','known_tokens','local_observations')}
        coverage.append(dict(scope='ACTUAL_TRAIN_SAMPLER',method=arm['branch'],seed=arm['seed'],distinct_originals=len(seen),
            original_draws=len(draws),distinct_frames=len(frames),clean_corrupt_view_observations=view_reads,**totals,
            correlated_repeated_observations=True))
    if real:coverage.append(dict(scope='REAL_PROPOSALS',status=real['status'],generated=real['generated'],retained=real['retained'],liftable=real['liftable'],multiview=real['multiview'],matched=real['matched']))
    h2=optional(root/'holdout2/receipt.json')
    if h2:coverage.append(dict(scope='H2',status=h2['status'],original_objects=h2['original_objects'],variant_records=h2['variant_records'],families=h2['families'],novel_original_objects=h2['novel_original_objects'],novel_families=h2['novel_families']))
    if d1:coverage.extend(dict(scope='D1_'+role,target_fallbacks=n,GT_DIAGNOSTIC_ONLY=True) for role,n in d1['fallback_counts'].items())
    table('table3_coverage',coverage or [dict(scope='NO_COMPLETED_COVERAGE',status='PARTIAL_ACTIVE_STAGE')])
    return tables,displayed

def package(binding,folder):
    """Copy only task-owned compact evidence; never traverse data/feature caches."""
    root=Path(binding['output_root']);copied=[]
    names=['source_binding.json','dependency_manifest.json','run_registry.json','storage_preflight.json',
           'prepare.json','data_profile.json','split_manifest.json','class_split.json',
           'H2_plan.json','H2_pretraining_inventory.json','holdout2/features.json','real_proposals/features.json',
           'real_proposals/generator_lock.json','real_proposals/generator_raw.json','real_proposals/input_lock.json','real_proposals/output_lock.json','nomination_seed17.json','nomination_seed29.json',
           'predictor_lock.json','D1.log','sparse_parity.log','training/restarts.json']
    # The current invocation cannot observe its own exit while generating reports.
    # Preserve previous attempts under observations/, and archive the final parent
    # observation plus stdout under terminal/ only after the actual process exits.
    for name in ('execution_last.json','execution_observed.json'):
        stale=folder/name
        if stale.exists():stale.unlink()
    files=[root/n for n in names if (root/n).is_file()]
    for name in ('prior_audit','diagnostic_D1','diagnostics','recognition','export','decisions','phases','phase_history','failures'):
        if (root/name).exists():files.extend(p for p in (root/name).rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.pt','.log'))
    files += [p for p in (root/'engineering').glob('*.json')]
    files += [p for p in (root/'observations').glob('*.json')]
    for name in ('resume_boundary_fix','confirmation_interface_fix'):
        restart=root/'history'/name
        if restart.exists():files.extend(p for p in restart.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jsonl','.log'))
    for name in ('confirmation','holdout2','real_proposals','evaluation','pools','predictions'):
        for relative in (name+'/receipt.json',name+'/store.json',name+'/summary.json',name+'/diagnostics.json',name+'/recognition_summary.json'):
            p=root/relative
            if p.is_file():files.append(p)
    for name in ('holdout2/recognition','holdout2/diagnostics','real_proposals/scores','real_proposals/intervention_scores'):
        if (root/name).exists():files.extend(p for p in (root/name).rglob('*') if p.is_file() and p.suffix in ('.json','.npz'))
    for p in sorted((root/'training').glob('seed*/*/*')):
        if p.is_file() and (p.suffix in ('.json','.npz') or p.name=='updates.jsonl'):files.append(p)
    for p in sorted((root/'training').glob('seed*/*')):
        if p.is_file() and p.suffix in ('.json','.npz'):files.append(p)
    index=ConsumptionIndex()
    for source in sorted(set(files)):
        relative=source.relative_to(root);dest=folder/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if source.name=='updates.jsonl':
            dest=dest.with_suffix('.jsonl.gz')
            with dest.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:gz.write(source.read_bytes())
        else:shutil.copyfile(source,dest)
        if dest.stat().st_size>=100*2**20:raise ValueError('Compact artifact exceeds normal Git file limit: '+str(dest))
        copied.append(dict(source=str(source),relative_path=str(dest.relative_to(folder)),**{k:v for k,v in index.identity(dest).items() if k!='path'}))
    return write(folder/'artifact_index.json',dict(status='COMPLETE',files=copied,raw_scans_GT_dense_and_foundation_weights_included=False))

def report(binding):
    root=Path(binding['output_root']);folder=REPO/binding['specification']['artifact_root'];folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();lock=optional(root/'predictor_lock.json') or {}
    evaluation=optional(root/'evaluation/store.json');missing,updates,arms=completion(binding,lock,evaluation)
    selection=map_selection(binding,evaluation,lock)
    if missing:status='PARTIAL_ACTIVE_STAGE'
    elif lock['DEV_nominee'] is None:status='COMPLETE_NO_FG_SIGNAL'
    elif not lock['activated_full_path']:status='COMPLETE_FG_NOT_REPEATED'
    else:status=selection['status']
    selection=write(root/'selection.json',dict(selection,scientific_status=status))
    tables,dev=table_families(binding,folder,lock,evaluation,arms)
    claims=write(root/'claims.json',dict(status=status,ordinary_frozen_FC_base=True,new_geometry_claim=False,
        online_FPS_claim=False,independent_map_generalization_claim=False,R_failure_cause_identified=False,
        FG2_vs_FG1='REQUIRES_MATCHED_GROUPING_CONTRAST',FG4_vs_FG3='REQUIRES_MATCHED_SUPERVISION_CONTRAST',
        routing_mechanism='REQUIRES_ACCURACY_AND_MATCHED_MASS_NORM_CONTRASTS',D1='GT_DIAGNOSTIC_ONLY_NOT_UPPER_BOUND',
        old_H='EXPOSED_POST_LOCK_REGRESSION',H2='CONDITIONAL_PROPOSAL_RECOGNITION_NOT_NEW_MAP_AP',
        deployment='N0_UNCHANGED',missing_active_requirements=missing))
    packaged=package(binding,folder)
    for name in ('selection.json','claims.json'):shutil.copyfile(root/name,folder/name)
    rows=(evaluation or {}).get('scene_method_coverage',0);pools=(evaluation or {}).get('full_cohort_pool_coverage',0)
    result=write(root/'result_store.json',dict(status=status,source_binding=binding['identity'],parent=binding['parent_store_identity'],
        predictor_lock=lock.get('identity'),scientific_optimizer_updates=updates,
        replayed_optimizer_updates=sum(r.get('discarded_uncheckpointed_updates',0) for r in arms),selected_heads=len(lock.get('heads',{})),
        engineering_discarded_updates=(optional(root/'engineering/receipt.json') or {}).get('discarded_updates',0),
        scene_method_coverage=rows,full_cohort_pool_coverage=pools,selection=selection['identity'],claims=claims['identity'],
        tables=tables,artifact_index=packaged['identity'],roundtrip=(optional(root/'export/roundtrip.json') or {}).get('identity'),
        missing_active_requirements=missing,deployment='N0_UNCHANGED',artifact_root=str(folder),
        metric_unit='RAW_FRACTIONS_DISPLAY_PERCENTAGES_WITH_EXPLICIT_SUFFIX',raw_scans_GT_dense_and_foundation_weights_included=False,
        elapsed_seconds=time.perf_counter()-started))
    shutil.copyfile(root/'result_store.json',folder/'result_store.json')
    docs=REPO/'docs/paper/static_ovmap';table=(folder/'tables/table2_DEV_selection.md').read_text()
    nominee=lock.get('DEV_nominee');qualified=[f"{r['branch']} (seed{r['seed']}, step{r['selected_step']})" for r in arms if r['qualified']]
    findings=(f"完成状态：**{status}**。科学优化 {updates:,} 步；选定小型头 {len(lock.get('heads',{}))} 个；新整图评分 {rows} 行 / {pools} 个池。\n\n"
              f"已记录的重复优化步数：{sum(r.get('discarded_uncheckpointed_updates',0) for r in arms)}，与科学更新分开统计。\n\n"
              f"DEV 固定提名：`{nominee}`。通过 DEV 门槛：{', '.join(qualified) or '无'}。部署保持 `N0_UNCHANGED`。\n\n")
    if status=='COMPLETE_NO_FG_SIGNAL':findings+='四个主轮方法均未通过固定 DEV 门槛；按预注册协议不触发第二种子、整图、H2 和真实提案扩展。诊断与全部选定权重仍已保存。\n\n'
    elif status=='COMPLETE_FG_NOT_REPEATED':findings+='主轮提名未在第二种子重复通过；保留全部八个方法结果，不切换提名来触发整图扩展。\n\n'
    elif missing:findings+='尚缺激活阶段证据：'+', '.join(missing)+'。当前结果不得表述为完整科学实验。\n\n'
    findings += 'DEV 数字以百分比显示，CE/NLL 保留原单位；69 个原始对象的四种条件相关，不能视作276个独立样本。\n\n'+table
    if evaluation:
        findings += '\n\n官方整图有序池化结果（%，不平均场景 AP）：\n\n'+(folder/'tables/table1_whole_map.md').read_text()
        findings += f"\n整图研究候选：`{selection.get('EXPOSED_BENCHMARK_CANDIDATE') or 'NONE'}`。固定 DEV 提名与曝光整图候选分开记录；第二种子的完整结果见 `table1_whole_map_repeat`。\n"
    for name,relative in [('H2','holdout2/receipt.json'),('真实提案','real_proposals/receipt.json')]:
        confirmation=optional(root/relative)
        if confirmation:
            if name=='H2':
                detail=f"原始对象 {confirmation.get('original_objects')}，novel 对象 {confirmation.get('novel_original_objects')}，novel 家族 {confirmation.get('novel_families')}"
            else:
                detail=f"生成 {confirmation.get('generated')}、保留 {confirmation.get('retained')}、多视图支持 {confirmation.get('multiview')}、有效 GT 匹配 {confirmation.get('matched')}"
            findings += f"\n{name}：`{confirmation['status']}`；{detail}。这是提案识别诊断，不是独立整图 AP。\n"
    findings += '\n三类表格见 [artifacts](../../../artifacts/static_ovmap/frozen_fc_routing_v1/tables)。整图未触发时，仅展示明确标注的继承基线；新方法单元格为空。\n'
    (docs/'FROZEN_FC_ROUTING_RESULTS.md').write_text('# Frozen FC routing results\n\n'+findings)
    command=(f"{binding['specification']['controller_python_hint']} -u scripts/evaluation/run_ovimap_frozen_fc_routing.py "
        f"--spec configs/static_ovmap/frozen_fc_routing_v1.json --parent-root {binding['pr_parent_root']} "
        f"--output-root {binding['logical_root']} --gpu {binding['gpu']} --phase all --resume")
    handoff=f'''# Frozen FC routing handoff

Task: `{binding['protocol']}`; base `{binding['base_commit']}`; fixed spec in `configs/static_ovmap/frozen_fc_routing_v1.json`.
Current science: `{status}`. Root `{root}`; worktree `{REPO}`; parent PR `{binding['pr_parent_root']}`; LR `{binding['lr_parent_root']}`; source-update `{binding['lr_source_root']}`.
Controller `{binding['specification']['controller_python_hint']}`; FC `{binding['FC_python']}`; single GPU `{binding['gpu']}` queued through the inherited resource file. Existing FP32 environments are unchanged; CUDA worked despite the pre-existing NVML warning.

```bash
{command}
```

Use the same command with any fixed `--phase` for recovery. Source, fixed split/bank/spec and training-code identities bind resume; preserve original failed attempts. `last.pt` keeps optimizer, sampler index and Python/NumPy/Torch/CUDA RNG. Saved scalar curves contain every completed update; replayed uncheckpointed updates are separate. Do not change scientific settings to recover a negative metric.

Training: four fresh identically initialized216,194-parameter heads, seed17,2000 updates each; all four seed29 only on a seed17 qualification. Exact inherited class-uniform original sampling,2/4/8 prefixes, first zero-LR update, FP32 AdamW and eta ramp. The explicit seven-key JSON whitelist governs the prose's erroneous count of eight. Base loss retains the formal0.1 student consistency term; no additional consistency/preservation/correspondence objective.

Ordinary FC region vectors use hard-mask/area-v2 fallback. Sparse inputs, when used, are exact tensors produced by the original loader, validated on real objects for bitwise inputs/outputs and input gradients. TRAIN/DEV/oldH/H2 use registered depth-grid RGB; regression banks retain their original preprocessing. No MA/R/learned768x768 value projection is an inference dependency.

Load interface: `static_ovmap.frozen_fc_routing.export.load_exported(registry_file, name, device, FC_identity=..., text_identity=...)`. Registry at `artifacts/static_ovmap/frozen_fc_routing_v1/export/weights_registry.json`; weights use relative paths and strict state loading. Reconstruct with the same ordered200-row FC text table and FrozenFC.phi. Raw scans/GT maps/dense caches/frozen backbone weights remain external. Artifact index maps compact source receipts to published files.

Completed science updates: {updates}; new scene rows/pools: {rows}/{pools}; missing active requirements: {missing}. All applicable selected heads, full scalar curves, checkpoint metrics, compact embeddings/cosines, decisions, D0/D1 and post-lock mass/norm diagnostics are archived. D1 is GT_DIAGNOSTIC_ONLY and cannot nominate a model; undefined norm pairs are masked on both paired sides while ordinary FG4 retains full support. Real proposal purity is NOT_COMPUTED without exact independent site targets.

Tables have MD/CSV/JSON/LaTeX and explicit percent columns; machine evaluation receipts retain fractions. Recognition is correlated over conditions and repeated views. DEV nomination, benchmark-exposed exploratory recommendation and same-architecture repeat remain distinct. H2 has fixed names and is proposal recognition, not independent map AP. Old H is exposed. Runtimes have explicit resident/cache boundaries; resumed/nested stage clocks must not be summed into a cold end-to-end total, and no online FPS claim is made.

The actual all --resume process exit is observed by an external parent. After that process exits, terminal/ archives its final execution receipt, parent observation and complete stdout, with a separate terminal/index.json. Prior observations remain in observations/; they are attempt history, not the final exit proof. Publication/final.json is written externally only after normal push and full local/remote SHA equality. No commit embeds its own future SHA.
'''
    (docs/'FROZEN_FC_ROUTING_HANDOFF.md').write_text(handoff)
    (docs/'FROZEN_FC_ROUTING_SELECTION.md').write_text('# Frozen FC selection\n\n'+json.dumps(selection,indent=2,ensure_ascii=False)+'\n\nA/M/C use69 DEV base originals with full200 competitors. Qualification requires A gain>=0.005, nondecreasing M and C, and positive raw net corrections. Checkpoints250/500/1000/1500/2000 rank by A/M/C descending, base160 CE ascending, then earlier step. Step0 cannot qualify. Same-step contrasts and independently selected-step contrasts are separate. Seed29 cannot replace the seed17 architecture. Maps/H2/runtime never choose checkpoints.\n')
    (docs/'FROZEN_FC_ROUTING_CLAIMS.md').write_text('# Frozen FC claim boundaries\n\n'+json.dumps(claims,indent=2,ensure_ascii=False)+'\n\nFG4 versus FG3 controls additional membership supervision. FG4 versus matched mass and matched delta norm controls different aspects of routing. A damaged intervention alone does not demonstrate superiority over FC or better maps. The convex-mixture nonzero bound is not an argmax/AP preservation guarantee.\n')
    return result
