"""Evidence-derived tables and release; no fabricated optional-stage numbers."""
import gzip
from functools import cmp_to_key
import json
import shutil
from pathlib import Path
import time
import numpy as np
from .common import REPO,ConsumptionIndex,canonical_digest,read,verified,write
from .tabulate import write_table

METRICS=('apall','ap50','ap25','miou','macc')

def pct(value):return None if value is None else 100*float(value)

def micro(records,condition=None):
    entries=[]
    for r in records:
        for i,(p,nll) in enumerate(zip(r['predictions'],r['NLL'])):
            if condition is None or i==condition:entries.append((r['key'],r['class_id'],p,nll))
    if not entries:return dict(objects=0,records=0,top1_pct=None,macro_pct=None,NLL=None)
    classes=sorted({e[1] for e in entries})
    return dict(objects=len({e[0] for e in entries}),records=len(entries),
         top1_pct=pct(np.mean([e[1]==e[2] for e in entries])),
         macro_pct=pct(np.mean([np.mean([e[1]==e[2] for e in entries if e[1]==c]) for c in classes])),
         NLL=float(np.mean([e[3] for e in entries])))

def map_selection(binding,store,lock):
    if not lock['activated_full_path']:
        return dict(status='NOT_TRIGGERED_NO_FOUNDATION',DEV_NOMINEE=lock['DEV_nominee'],
               EXPOSED_BENCHMARK_CANDIDATE=None,repeat_status=lock['status'],research_retained='G1',deployment='N0_UNCHANGED')
    pools=store['pooled_metrics'];gates={};main=[]
    for method in store['methods']:
        if method in ('PR00_D2','PR01_G1','PR02_FC8'):continue
        checks={}
        for cohort in ('replica8','scannet_cf18'):
            candidate=pools[cohort][method]['metrics'];g1=pools[cohort]['PR01_G1']['metrics']
            checks[cohort]={k:candidate[k]>=g1[k]-1e-10 for k in METRICS}
        cf=pools['scannet_cf18'][method]['metrics'];d2=pools['scannet_cf18']['PR00_D2']['metrics']
        checks['CF_D2']=dict(APall_gt=cf['apall']>d2['apall']+1e-10,AP50_ge=cf['ap50']>=d2['ap50']-1e-10)
        passed=all(v for c in checks.values() for v in c.values())
        gain=cf['apall']-d2['apall'];gates[method]=dict(passed=passed,checks=checks,CF_APall_gain_fraction=gain,
                                                    material_CF_gain=gain>=.001-1e-10)
        if passed and not method.startswith('R29_'):
            ds=[pools[c][method]['metrics']['apall']-pools[c]['PR01_G1']['metrics']['apall'] for c in pools]
            mi=[pools[c][method]['metrics']['miou']-pools[c]['PR01_G1']['metrics']['miou'] for c in pools]
            main.append((method,float(np.mean(ds)),min(ds),float(np.mean(mi))))
    order=['PR_RSTAR',*[m['id'] for m in binding['specification']['G_methods']]]
    def compare(a,b):
        for av,bv in zip(a[1:],b[1:]):
            if abs(av-bv)>1e-10:return -1 if av>bv else 1
        return order.index(a[0])-order.index(b[0])
    winner=sorted(main,key=cmp_to_key(compare))[0][0] if main else None
    repeat='R29_'+winner if winner else None
    return dict(status='TARGET_MET' if winner else 'NO_TARGET_GAIN',DEV_NOMINEE=lock['DEV_nominee'],
          EXPOSED_BENCHMARK_CANDIDATE=winner,benchmark_selection_exploratory=True,gates=gates,
          same_architecture_repeat=repeat,repeat_map_pass=gates[repeat]['passed'] if repeat else None,
          research_retained=winner or 'G1',deployment='N0_UNCHANGED',H2_picks_winner=False,latency_gate=False)

def tables(binding,folder,store,lock):
    root=Path(binding['output_root']);rows=[];identities={}
    columns=['cohort','method','seed','status','APall_pct','AP50_pct','AP25_pct','mIoU_pct','mAcc_pct']
    if lock['activated_full_path']:
        for cohort,pools in store['pooled_metrics'].items():
            for m,pool in pools.items():
                v=pool['metrics'];rows.append(dict(cohort=cohort,method=m,seed=29 if m.startswith('R29_') else 17 if m in lock['map_heads'] else None,
                      status='MEASURED_OFFICIAL_ORDERED_POOL',**dict(zip(columns[4:],[pct(v[k]) for k in METRICS]))))
    else:
        for cohort in binding['cohorts']:
            rows.append(dict(cohort=cohort,method='NEW_MAP_PATH',seed=None,status='NOT_TRIGGERED_NO_FOUNDATION',
                             **{k:None for k in columns[4:]}))
    identities['table1_whole_map']=write_table(folder/'tables','table1_whole_map',columns,[r for r in rows if r['seed']!=29])
    repeat_rows=[r for r in rows if r['seed']==29]
    if not repeat_rows:
        repeat_rows=[dict(cohort=c,method='REPEAT_MAP_PATH',seed=29,status='NOT_TRIGGERED_NO_FOUNDATION',
                          **{k:None for k in columns[4:]}) for c in binding['cohorts']]
    identities['table1_whole_map_repeat']=write_table(folder/'tables','table1_whole_map_repeat',columns,repeat_rows)
    columns=['scope','method','seed','step','trained_updates','status','A_pct','M_pct','C_pct','CE_base160','NLL_full200',
             'corrections','harm','net_corrections','original_objects','all_full200_top1_pct','heldout_top1_pct']
    rows=[]
    def proposal_row(scope,name,seed,step,updates,status,value):
        metrics=value['metrics'];records=value['records']
        return dict(scope=scope,method=name,seed=seed,step=step,trained_updates=updates,status=status,
                    A_pct=pct(metrics['A']),M_pct=pct(metrics['M']),C_pct=pct(metrics['C']),CE_base160=metrics['CE'],
                    NLL_full200=metrics['NLL'],corrections=metrics['corrections'],harm=metrics['harm'],net_corrections=metrics['net_corrections'],
                    original_objects=len(records),all_full200_top1_pct=micro(records)['top1_pct'],
                    heldout_top1_pct=micro([r for r in records if not r['base']])['top1_pct'])
    fc=verified(root/'recognition/dev_FC8.json');rows.append(proposal_row('DEV_BASE_SELECTION','FC8',None,None,0,'FROZEN_REFERENCE',fc))
    for name,head in lock['heads'].items():
        seed=head['seed'];branch=head['branch'];r=verified(root/f'training/seed{seed}'/branch/'receipt.json')
        value=verified(root/f'training/seed{seed}'/branch/f'dev_{r["selected_step"]:04d}.json')
        rows.append(proposal_row('DEV_BASE_SELECTION',name,seed,r['selected_step'],2000,'PASS_DEV_GATE' if r['foundation_pass'] else 'BEST_DIAGNOSTIC_NOT_QUALIFIED',value))
    for role,subfolder in [('OLD_H_EXPOSED','recognition/old_H'),('H2_NEW_FAMILY_PROPOSALS','holdout2/recognition')]:
        for p in sorted((root/subfolder).glob('*.json')):
            value=verified(p);name=p.stem;head=lock['heads'].get(name)
            rows.append(proposal_row(role,name,head['seed'] if head else None,
                      verified(root/f'training/seed{head["seed"]}'/head['branch']/'receipt.json')['selected_step'] if head else None,
                      2000 if head else 0,'POST_LOCK_NO_SELECTION',value))
    identities['table2_learning']=write_table(folder/'tables','table2_learning',columns,rows)
    parent=verified(root/'parent_audit/receipt.json');historical=[]
    for scope,methods in parent['TRAIN_DEV'].items():
        for name,row in methods.items():
            v=row['metrics']
            historical.append(dict(scope=scope,method=name,status='HISTORICAL_PARENT_READ_ONLY',
                A_pct=pct(v['A']),M_pct=pct(v['M']),C_pct=pct(v['C']),CE_base160=v['CE'],NLL_full200=v['NLL'],
                corrections=v['corrections'],harm=v['harm'],net_corrections=v['net_corrections'],
                original_objects=v['original_objects']))
    historical_columns=['scope','method','status','A_pct','M_pct','C_pct','CE_base160','NLL_full200','corrections','harm','net_corrections','original_objects']
    identities['table2_parent_audit']=write_table(folder/'tables','table2_parent_audit',historical_columns,historical)
    # Direct paired contrasts retain signs and separately report both seeds.
    pairs=[('R1_MINUS_R0','PR_R1_RESIDUAL','PR_R0_DIRECT'),('R2_MINUS_R0','PR_R2_KEEP','PR_R0_DIRECT'),
           ('R3_MINUS_R1','PR_R3_RESIDUAL_KEEP','PR_R1_RESIDUAL'),('R3_MINUS_R2','PR_R3_RESIDUAL_KEEP','PR_R2_KEEP')]
    pairs += [(m['id']+'_MINUS_FC8',m['id'],'FC8') for m in binding['specification']['R_methods']]
    if lock['activated_full_path']:
        pairs += [('SURFACE_MINUS_VIEW','PR_G2_SURFACE','PR_G1_VIEW'),('AUX_MINUS_SURFACE','PR_G3_AUX_ONLY','PR_G2_SURFACE'),
                  ('ROUTED_MINUS_AUX_ONLY','PR_G4_ROUTED','PR_G3_AUX_ONLY'),('CORRESP_MINUS_ROUTED','PR_G5_CORRESP','PR_G4_ROUTED')]
        pairs += [(m['id']+'_MINUS_RSTAR',m['id'],lock['Rstar']) for m in binding['specification']['G_methods']]
        pairs += [(m['id']+'_MINUS_FC8',m['id'],'FC8') for m in binding['specification']['G_methods']]
    contrasts=[]
    by={(r['scope'],r['method']):r for r in rows}
    for scope in ('DEV_BASE_SELECTION','OLD_H_EXPOSED','H2_NEW_FAMILY_PROPOSALS'):
        for label,a,b in pairs:
            for prefix in ('','R29_'):
                second=b if b=='FC8' else prefix+b
                if (scope,prefix+a) not in by or (scope,second) not in by:continue
                aa=by[(scope,prefix+a)];bb=by[(scope,second)]
                contrasts.append(dict(scope=scope,contrast=label,seed=29 if prefix else 17,first=prefix+a,second=second,
                    first_step=aa['step'],second_step=bb['step'],delta_A_pp=None if aa['A_pct'] is None else aa['A_pct']-bb['A_pct'],
                    delta_M_pp=None if aa['M_pct'] is None else aa['M_pct']-bb['M_pct'],delta_C_pp=None if aa['C_pct'] is None else aa['C_pct']-bb['C_pct'],
                    delta_top1_pp=None if aa['all_full200_top1_pct'] is None else aa['all_full200_top1_pct']-bb['all_full200_top1_pct'],
                    delta_NLL=None if aa['NLL_full200'] is None else aa['NLL_full200']-bb['NLL_full200']))
    cols=['scope','contrast','seed','first','second','first_step','second_step','delta_A_pp','delta_M_pp','delta_C_pp','delta_top1_pp','delta_NLL']
    identities['table2_direct_pairs']=write_table(folder/'tables','table2_direct_pairs',cols,contrasts)
    columns=['record_type','scope','method','seed','condition','stratum','objects','records','top1_pct','macro_pct','equal_GT_top1_pct','NLL','mean_margin',
             'corrections','harm','null','negative_mass','target_mass','unknown_mass','negative_norm_proxy',
             'positive_tokens','negative_tokens','unknown_tokens','count','seconds','allocated_GiB','reserved_GiB','status']
    robust=[]
    def add(**values):robust.append({c:values.get(c) for c in columns})
    observed=set(verified(root/'prepare.json')['observed_positive_base_ids'])
    for role,subfolder in [('TRAIN','recognition/train'),('DEV_ALL_POST_LOCK','recognition/dev'),('OLD_H_EXPOSED','recognition/old_H'),('H2','holdout2/recognition')]:
        files=sorted((root/subfolder).glob('*.json'))
        if not files:continue
        teacher=verified(root/subfolder/'FC8.json');ref={r['key']:r for r in teacher['records']}
        for condition in ('truncate','append','truncate_append'):
            add(record_type='CORRUPTION_NO_OP_COVERAGE',scope=role,method='COMMON_PROPOSAL_BANK',condition=condition,
                objects=len(ref),count=sum(condition in r['no_op_conditions'] for r in ref.values()),
                status='ORIGINAL_OBJECT_DENOMINATOR_CORRELATED_CONDITIONS')
        for p in files:
            value=verified(p);name=p.stem;head=lock['heads'].get(name);seed=head['seed'] if head else None
            for ci,condition in enumerate(('clean','truncate','append','truncate_append','all_conditions')):
                for group in ('all','teacher_correct','teacher_wrong','observed_base','zero_positive_base','heldout'):
                    records=[r for r in value['records'] if group=='all' or
                         (group=='teacher_correct' and ref[r['key']]['predictions'][0]==r['class_id']) or
                         (group=='teacher_wrong' and ref[r['key']]['predictions'][0]!=r['class_id']) or
                         (group=='observed_base' and r['base'] and r['class_id'] in observed) or
                         (group=='zero_positive_base' and r['base'] and r['class_id'] not in observed) or
                         (group=='heldout' and not r['base'])]
                    metric=micro(records,ci if ci<4 else None);fix=harm=0
                    for r in records:
                        for i in ([ci] if ci<4 else range(4)):
                            pc=r['predictions'][i]==r['class_id'];tc=ref[r['key']]['predictions'][i]==r['class_id']
                            fix+=int(pc and not tc);harm+=int(tc and not pc)
                    add(record_type='PROPOSAL_RECOGNITION_MICRO',scope=role,method=name,seed=seed,condition=condition,stratum=group,
                        **metric,corrections=fix,harm=harm,
                        mean_margin=float(np.mean([r['margins'][i] for r in records for i in ([ci] if ci<4 else range(4))])) if records else None,
                        status='MEASURED' if records else 'NO_SUPPORT_NULL_METRICS')
    detail=verified(root/'diagnostics/interventions.json')
    for key,r in detail['records'].items():
        if 'group_count' in r:
            add(record_type='ROUTE_FIXED_DEV',scope=key,method=key.rsplit('|',1)[-1],
                null=r['null'],negative_mass=r['negative_coefficient_mass'],target_mass=r['target_coefficient_mass'],
                unknown_mass=r['unknown_coefficient_mass'],negative_norm_proxy=r['negative_coefficient_value_norm_proxy'],
                positive_tokens=r['positive_tokens'],negative_tokens=r['negative_tokens'],unknown_tokens=r['unknown_tokens'],
                count=r['group_count'],status='MEASURED_PROXY')
            for tag,value in r['interventions'].items():
                add(record_type='ROUTE_INTERVENTION',scope=key,method=key.rsplit('|',1)[-1],stratum=tag,
                    null=value.get('null'),count=int(value['prediction']==r['truth']) if 'prediction' in value else None,
                    status='FIXED_DIAGNOSTIC_NOT_CONTENDER')
            value=r['wrong_correspondence']
            add(record_type='WRONG_CORRESPONDENCE',scope=key,method=key.rsplit('|',1)[-1],
                count=int(value['prediction']==r['truth']),status='FIXED_DIAGNOSTIC_NOT_CONTENDER')
        elif '_G' in key.rsplit('|',1)[-1]:
            add(record_type='LOCAL_MISSING_SUPPORT',scope=key,method=key.rsplit('|',1)[-1],count=0,
                status='EXACT_FROZEN_BASE_NO_LOCAL_GROUPS')
    for seed in (17,29):
        for p in sorted((root/f'training/seed{seed}').glob('*/receipt.json')):
            r=verified(p)
            add(record_type='TRAINING_COMPUTE',scope='OPTIMIZER_PHASE_AND_FIVE_DEV_VALIDATIONS',method=r['branch'],seed=seed,count=r['completed_steps'],
                seconds=r['elapsed_seconds'],allocated_GiB=r['peak_allocated_bytes']/2**30,reserved_GiB=r['peak_reserved_bytes']/2**30,status=r['status'])
    teachers=verified(root/'teachers/manifest.json')
    for prefix,count in teachers['correct_by_prefix'].items():
        add(record_type='TRAIN_TEACHER_CORRECT',scope='PREFIX_'+prefix,method='FC',objects=teachers['denominator_per_prefix'],count=count,status='MEASURED_TRAIN_ONLY')
    for p in sorted((root/'phases').glob('*.json')):
        phase=verified(p)
        add(record_type='OBSERVED_STAGE_WALL',scope='LAST_COMPLETED_STAGE_RECEIPT_NESTED_CLOCK_DO_NOT_SUM_WITH_ARM_CLOCKS',method=p.stem,
            seconds=phase.get('stage_elapsed_seconds'),status=phase['status'])
    for p in sorted((root/'history').glob('*/phases/*.json')):
        phase=verified(p)
        add(record_type='OBSERVED_PRIOR_STAGE_WALL',scope=p.parents[1].name+'_NESTED_CLOCK_DO_NOT_SUM',method=p.stem,
            seconds=phase.get('stage_elapsed_seconds'),status=phase['status'])
    for p in sorted((root/'history').glob('*/observer_status.json')):
        if verified(p).get('actual_original_cli_exit_code') is None and p.with_name('execution_last.json').exists():
            previous=verified(p.with_name('execution_last.json'))
            add(record_type='CONTROLLER_REPORTED_PRIOR_WALL',scope='NESTED_CLOCK_EXIT_NOT_OBSERVED',method=p.parent.name,
                seconds=previous.get('elapsed_seconds'),status=previous['status'])
    observed_attempts=set()
    for p in sorted((root/'history').glob('*/execution_observed.json')):
        observer=p.parent/'observer_status.json'
        if observer.exists() and verified(observer).get('actual_original_cli_exit_code') is None:continue
        attempt=verified(p)
        if attempt['identity'] in observed_attempts:continue
        observed_attempts.add(attempt['identity'])
        add(record_type='OBSERVED_PRIOR_CLI_WALL',scope=p.parent.name,seconds=attempt['elapsed_seconds'],status=attempt['status'])
    if (root/'recognition/regression.json').exists():
        inference=verified(root/'recognition/regression.json')
        add(record_type='MAP_HEAD_FORWARD',scope=inference['timing_boundary'],objects=inference['original_objects'],
            count=inference['head_forwards'],seconds=inference['selected_head_forward_seconds'],
            allocated_GiB=inference['peak_allocated_bytes']/2**30,reserved_GiB=inference['peak_reserved_bytes']/2**30,status='MEASURED_CONDITIONAL')
        add(record_type='MAP_INPUT_LOADING_AND_FROZEN_POOLING',scope=inference['timing_boundary'],
            seconds=inference['loader_and_frozen_pooling_seconds'],status='MEASURED_CONDITIONAL')
        payload=verified(root/'predictions/summary.json')
        add(record_type='MAP_PAYLOAD_CONSTRUCTION',scope='ALL_FIXED_G1_OUTPUTS',count=payload['logical_records'],
            seconds=payload['payload_construction_seconds'],status='MEASURED_CONDITIONAL')
    for stage in ('real_proposals','holdout2'):
        p=root/stage/'receipt.json'
        if p.exists():
            r=verified(p);add(record_type='TRANSFER_COVERAGE',scope=stage,objects=r.get('original_objects'),records=r.get('variant_records'),
                   count=r.get('matched'),seconds=r.get('elapsed_seconds'),status=r['status'])
            if stage=='holdout2' and r['status']!='PARTIAL_CONFIRMATION_BLOCKED':
                add(record_type='H2_NOVEL_SUPPORT',scope='NOVEL_ORIGINAL_OBJECTS',count=r['novel_original_objects'],status=r['novel_status'])
                add(record_type='H2_NOVEL_SUPPORT',scope='NOVEL_FAMILIES',count=r['novel_families'],status=r['novel_status'])
                feature=verified(root/'holdout2/features.json')
                add(record_type='PHYSICAL_FC_IMAGE_ENCODINGS',scope='H2',count=feature['physical_encodings'],status='MEASURED')
            if stage=='real_proposals' and r['status']=='COMPLETE':
                feature=verified(root/'real_proposals/features.json');generator=verified(root/'real_proposals/generator_lock.json')
                add(record_type='PHYSICAL_FC_IMAGE_ENCODINGS',scope='REAL_PROPOSAL_CONTEXT',count=feature['new_FC_encodings'],status='MEASURED')
                for key,count in generator['counts'].items():
                    add(record_type='SAM2_GENERATOR_CALLS',scope=key,count=count,status='MEASURED')
                for count in ('generated','retained','liftable','multiview','matched'):
                    add(record_type='REAL_PROPOSAL_FUNNEL',scope='DEV_GT_FREE_SAM2',stratum=count,count=r[count],status='MEASURED_COMMON_COVERAGE')
                for name,quality in r['metrics'].items():
                    for q,metric in quality.items():
                        add(record_type='REAL_PROPOSAL_RECOGNITION',scope='DEV_GT_FREE_SAM2',method=name,stratum=q,
                            objects=metric['unique_GT'],records=metric['proposals'],top1_pct=pct(metric['proposal_top1']),
                            equal_GT_top1_pct=pct(metric['equal_GT_top1']),status='PROPOSAL_AND_EQUAL_GT_AVERAGES')
        else:add(record_type='TRANSFER_COVERAGE',scope=stage,status='NOT_TRIGGERED_NO_FOUNDATION' if not lock['activated_full_path'] else 'MISSING_REQUIRED_OUTCOMES')
    identities['table3_robustness_compute']=write_table(folder/'tables','table3_robustness_compute',columns,robust)
    summary=[]
    for name in ['FC8',*lock['heads']]:
        r=verified(root/'recognition/dev'/(name+'.json'));values=r['records']
        summary.append(dict(method=name,original_objects=len(values),clean_top1_pct=micro(values,0)['top1_pct'],
            truncate_top1_pct=micro(values,1)['top1_pct'],append_top1_pct=micro(values,2)['top1_pct'],
            both_top1_pct=micro(values,3)['top1_pct'],heldout_all_conditions_top1_pct=micro([v for v in values if not v['base']])['top1_pct']))
    identities['table3_DEV_robustness']=write_table(folder/'tables','table3_DEV_robustness',list(summary[0]),summary)
    compute=[dict(method=r['method'],seed=r['seed'],updates=r['count'],seconds=r['seconds'],
                  allocated_GiB=r['allocated_GiB'],reserved_GiB=r['reserved_GiB']) for r in robust if r['record_type']=='TRAINING_COMPUTE']
    identities['table3_training_compute']=write_table(folder/'tables','table3_training_compute',list(compute[0]),compute)
    registry=verified(root/'export/weights_registry.json')
    parameters=[dict(method='FC_FROZEN_BACKBONE',trained_head_parameters=0,frozen_initial_parameters=0,
                     frozen_base_parameters=0,frozen_FC_parameters=registry['frozen_FC_parameters'],
                     nonzero_gradient_parameters=None,gradient_observation_step=None)]
    for name,r in registry['heads'].items():
        parameters.append(dict(method=name,trained_head_parameters=r['trainable_stored_parameters'],
            frozen_initial_parameters=r['frozen_initial_parameters'],frozen_base_parameters=r['frozen_base_parameters'],
            frozen_FC_parameters=registry['frozen_FC_parameters'],
            nonzero_gradient_parameters=r['nonzero_gradient_parameters_at_last_logged_update'],
            gradient_observation_step=r['last_logged_gradient_step']))
    identities['table3_parameters']=write_table(folder/'tables','table3_parameters',list(parameters[0]),parameters)
    return identities,contrasts

def report(binding):
    root=Path(binding['output_root']);started=time.perf_counter();lock=verified(root/'predictor_lock.json')
    store=verified(root/'evaluation/store.json');roundtrip=verified(root/'export/roundtrip.json')
    registry=verified(root/'export/weights_registry.json')
    if roundtrip['predictor_lock']!=lock['identity'] or roundtrip['registry']!=registry['identity']:
        raise ValueError('Selected inference roundtrip proof differs')
    if lock['activated_full_path'] and (store['scene_method_coverage'],store['full_cohort_pool_coverage'])!=(390,30):
        raise ValueError('Activated whole-map scientific coverage missing')
    folder=REPO/'artifacts/static_ovmap/preservation_routing_v1';folder.mkdir(parents=True,exist_ok=True)
    selected=write(root/'selection.json',map_selection(binding,store,lock));index=ConsumptionIndex()
    copied=[]
    # These directories contain compact scores/decisions, never source scans, masks or feature grids.
    for directory in ('recognition','decisions','diagnostics','export','phases','failures','validation','holdout2/recognition',
                      'holdout2/failed_attempts','real_proposals/scores','parent_audit/recognition'):
        source=root/directory
        if source.exists():
            for p in source.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.npz','.pt'):
                    dest=folder/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
                    copied.append(index.identity(dest))
    for p in (root/'history').glob('*/execution_observed.json'):
        observer=p.parent/'observer_status.json'
        if observer.exists() and verified(observer).get('actual_original_cli_exit_code') is None:
            dest=folder/observer.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(observer,dest)
            if p.with_name('execution_last.json').exists():shutil.copyfile(p.with_name('execution_last.json'),dest.with_name('execution_last.json'))
            continue
        dest=folder/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for p in (root/'history').glob('*/phases/*.json'):
        dest=folder/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for p in (root/'history').glob('*/stage_archive.json'):
        dest=folder/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for n in ('source_binding.json','dependency_manifest.json','method_registry.json','class_split.json','split_manifest.json','prepare.json',
              'H2_plan.json','H2_pretraining_inventory.json','predictor_lock.json','R_nomination_seed17.json','R_nomination_seed29.json','G_nomination_seed17.json','G_nomination_seed29.json',
              'selection.json','teachers/manifest.json','teachers/train_prefixes.npz','parent_audit/receipt.json','engineering/receipt.json','engineering/two_scene.json',
              'holdout2/receipt.json','holdout2/features.json','real_proposals/receipt.json','real_proposals/features.json',
              'real_proposals/generator_lock.json','real_proposals/input_lock.json','real_proposals/output_lock.json'):
        p=root/n
        if p.exists():
            dest=folder/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);copied.append(index.identity(dest))
    for p in (root/'parent_audit/curves').rglob('*'):
        if p.is_file():
            dest=folder/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    integration=verified(root/'engineering/two_scene.json')
    for scene,row in integration['scenes'].items():
        shutil.copyfile(row['scoring_receipt']['evaluation_receipt'],folder/'engineering'/('scorer_'+scene+'.json'))
    for p in (root/'training').rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if p.name=='updates.jsonl':
            dest=(folder/rel).with_suffix('.jsonl.gz');dest.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(dest,'wb') as stream:stream.write(p.read_bytes())
        elif p.suffix in ('.json','.npz'):
            dest=folder/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    if lock['activated_full_path']:
        for directory in ('evaluation','pools'):
            for p in (root/directory).rglob('*.json'):
                dest=folder/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        bundles={}
        for row in store['scene_metrics']:
            key=row['evaluation_identity']
            if key in bundles:continue
            source=Path(row['evaluation_receipt']);dest=folder/'scoring_bundles'/key;dest.mkdir(parents=True,exist_ok=True)
            files={}
            for name in ('receipt.json','matches.json.gz','trace.json.gz'):
                p=source.with_name(name)
                if not p.is_file():raise FileNotFoundError('Actual released scoring bundle missing: '+str(p))
                shutil.copyfile(p,dest/name)
                files[name]=dict(index.identity(dest/name),relative_path=str((dest/name).relative_to(folder)))
            bundles[key]=files
        write(folder/'scoring_bundles/manifest.json',dict(status='COMPLETE',bundles=bundles,
              scene_methods={r['scene']+'|'+r['method']:r['evaluation_identity'] for r in store['scene_metrics']},
              raw_prediction_masks_and_GT_maps_included=False))
    table_ids,contrasts=tables(binding,folder,store,lock)
    successful_updates=sum(verified(p)['completed_steps'] for p in (root/'training').glob('seed*/*/receipt.json'))
    expected=30000 if lock['activated_full_path'] else 8000 if lock['status']=='COMPLETE_NO_2D_FOUNDATION' else 10000
    if successful_updates!=expected:raise ValueError('Applicable fixed training budget incomplete')
    claims=write(root/'claims.json',dict(status=lock['status'],
        preservation_foundation_repeated=lock['activated_full_path'],
        R_factorial_contrasts_initially_single_seed=True,no_claim_all_three_innovations_universal=True,
        H_old_exposed=True,H2_new_family_proposal_not_independent_map=True,
        real_proposals_DEV_diagnostic_not_training=True,group_vector_norm_or_accuracy_not_monotonic=True,
        MECHANISM_EVIDENCE=dict(direct_pairs=contrasts,R_factorial='SINGLE_SEED_DEV_SELECTED_STEPS_MAY_DIFFER',
             G_pairs='BOTH_SEED_SIGNS_RETAINED' if lock['activated_full_path'] else 'NOT_TRIGGERED_NO_FOUNDATION',
             causal_or_universal_validation_claim=False),
        direct_pairs=contrasts,whole_map_selection=selected['identity'],deployment='N0_UNCHANGED'))
    shutil.copyfile(root/'claims.json',folder/'claims.json')
    h2=verified(root/'holdout2/receipt.json') if (root/'holdout2/receipt.json').exists() else None
    real=verified(root/'real_proposals/receipt.json') if (root/'real_proposals/receipt.json').exists() else None
    h2_text=('H2: NOT_TRIGGERED_NO_FOUNDATION.' if not lock['activated_full_path'] else
             'H2: mandatory outcomes are missing.' if not h2 else
             'H2: PARTIAL_CONFIRMATION_BLOCKED; missing outcomes remain null.' if h2['status']=='PARTIAL_CONFIRMATION_BLOCKED' else
             f"H2: {h2['original_objects']} originals; {h2['novel_original_objects']} heldout originals in {h2['novel_families']} families. Scope: {h2['status']}; novel scope: {h2['novel_status']}.")
    technical=lock['activated_full_path'] and (not h2 or h2['status']=='PARTIAL_CONFIRMATION_BLOCKED' or not real or real['status']!='COMPLETE')
    status='PARTIAL_CONFIRMATION_BLOCKED' if technical else 'SCIENCE_COMPLETE' if lock['activated_full_path'] else lock['status']
    result=write(root/'result_store.json',dict(status=status,source_binding=binding['identity'],parent=binding['parent_store_identity'],
           predictor_lock=lock['identity'],selection=selected['identity'],claims=claims['identity'],
           scientific_optimizer_updates=successful_updates,engineering_discarded_updates=20,
           selected_heads=len(registry['heads']),roundtrip=roundtrip['identity'],tables=table_ids,
           scene_method_coverage=store['scene_method_coverage'],full_cohort_pool_coverage=store['full_cohort_pool_coverage'],
           deployment='N0_UNCHANGED',metric_unit='FRACTION_IN_EVALUATION_PERCENT_IN_DISPLAY_TABLES',
           elapsed_seconds=time.perf_counter()-started,artifact_root=str(folder),raw_scans_GT_and_dense_features_included=False))
    shutil.copyfile(root/'result_store.json',folder/'result_store.json')
    docs=REPO/'docs/paper/static_ovmap';base='../../../artifacts/static_ovmap/preservation_routing_v1/'
    fc=verified(root/'recognition/dev_FC8.json')['metrics'];main=verified(root/'R_nomination_seed17.json')
    arm_lines=[]
    for name,metrics in main['arms'].items():
        arm_lines.append(f"| {name} | {metrics['step']} | {pct(metrics['A']):.2f} | {pct(metrics['M']):.2f} | {pct(metrics['C']):.2f} | {metrics['net_corrections']} |")
    (docs/'PRESERVATION_ROUTING_RESULTS.md').write_text(
        f"# Preservation routing results\n\nStatus: **{status}**. Deployment: `N0_UNCHANGED`.\n\n"+
        f"Actual scientific updates: {successful_updates}; discarded engineering updates:20. TRAIN:622 base-positive originals /74 positive categories; DEV:69 base originals /4 families. Exact parent24/4 split and160/40 class IDs retained.\n\n"+
        f"FC8 DEV A/M/C (%): {pct(fc['A']):.2f}/{pct(fc['M']):.2f}/{pct(fc['C']):.2f}.\n\n"+
        '| Arm | Selected step | A (%) | M (%) | C (%) | Net corrections |\n| --- | --- | --- | --- | --- | --- |\n'+'\n'.join(arm_lines)+'\n\n'+
        f"Rstar: `{main['Rstar']}`. Checkpoints0 are diagnostics and never nominees. Four R arms have matching fresh MA initialization and32000 original-item draws each. Teacher correctness uses TRAIN clean full200 FC of the same prefix, and never enters inference.\n\n"+
        f"[Table1: full maps]({base}tables/table1_whole_map.md) · [Seed29 map repeat]({base}tables/table1_whole_map_repeat.md) · [Table2: learning]({base}tables/table2_learning.md) · [Direct pairs]({base}tables/table2_direct_pairs.md) · [Table3: robustness/compute]({base}tables/table3_robustness_compute.md). JSON retains numeric types/full precision; displayed accuracies use percentages.\n\n"+
        ('Both R qualifications passed; five G architectures were trained in both seeds. Whole maps preserve G1 geometry/recovery/protected labels, replace only eligible F readouts, rebuild current-class ranks and use released ordered pooling.\n\n' if lock['activated_full_path'] else
         'The fixed foundation/repeat gate failed. G training, new whole-map expansion, H2 acquisition and SAM2 proposal generation were NOT_TRIGGERED. Null table cells are unmeasured, never zero accuracy. Selected diagnostic R heads, component curves and exposed oldH regression are retained.\n\n')+
        'OldH is exposed regression after nomination. TRAIN/DEV/H proposal conditions are correlated versions of original objects. TRAIN/DEV use registered depth-grid RGB; map inputs retain their inherited preprocessing domain. No independent map/geometry gain is inferred from proposal recognition.\n\n'+
        h2_text+'\n\n'+
        f"[Compact Table3: DEV corruptions]({base}tables/table3_DEV_robustness.md) uses all87 post-lock DEV objects with full200 competition, including18 heldout-class objects; these micro accuracies never select checkpoints. [Training compute]({base}tables/table3_training_compute.md) records optimizer work plus five DEV checks per arm. Detailed strata, no-op denominators and prior/latest nested stage clocks remain in the full Table3 supplement.\n\n"+
        'Scalar component curves and effective denominators are published under training/. Parent summed-loss/component availability is documented in parent_audit/receipt.json; unavailable historical CE/consistency values remain null.\n\n'+
        f"[Parent TRAIN/DEV audit]({base}tables/table2_parent_audit.md) separates the retained historical heads from this fresh experiment. [Stored parameter counts]({base}tables/table3_parameters.md) separate trained heads, frozen initial MA/base and the external FC backbone.\n\n"+
        'Runtime is recorded, has no qualification gate, and is not an online30FPS claim. Training clocks include DEV validation; conditional cached-feature inference and output construction have separate boundaries. Residual inference requires the explicitly exported frozen initial MA in addition to trainable MA and frozen FC.\n')
    (docs/'PRESERVATION_ROUTING_SELECTION.md').write_text(
        '# Preservation routing selection\n\n'+
        'DEV uses base-object labels, full200 competitors, equal-family A/C and equal-class M. Passing checkpoints require A>=FC8+.005, M/C nondecreasing and positive raw net corrections; step>=250. Ordering A/M/C descending, base160 CE ascending, earlier step, then R0/R1/R2/R3; tolerance1e-10.\n\n'+
        f"Main Rstar: `{main['Rstar']}`; terminal path: `{lock['status']}`.\n\n"+
        f"DEV_NOMINEE: `{selected['DEV_NOMINEE']}`. EXPOSED_BENCHMARK_CANDIDATE: `{selected['EXPOSED_BENCHMARK_CANDIDATE']}`. Research retained: `{selected['research_retained']}`. Deployment: `N0_UNCHANGED`.\n\n"+
        'Seed29 repeats only the preselected R architecture; if qualified, all five G arms repeat. It never changes the main nominated architecture. Whole-map gates require both cohorts all five metrics>=G1, CF APall>D2 and AP50>=D2; material CF APall gain0.001 fraction. Benchmark-picked winners are exploratory. H2, oldH, maps and runtime cannot admit a failed R or select a checkpoint.\n\n'+
        f"[Machine-readable selection]({base}selection.json), [R nominations]({base}R_nomination_seed17.json).\n")
    (docs/'PRESERVATION_ROUTING_CLAIMS.md').write_text(
        '# Preservation routing claim boundaries\n\n'+
        'Residual initialization equals same-input FC and actual first backward reaches MA through frozen phi. A within-group common quality shift cancels; fixed null routing permits total local mass to vanish. These arithmetic properties do not establish accuracy, causally explain all LR failure, or establish novelty.\n\n'+
        f"Repeated qualified2D foundation: `{lock['activated_full_path']}`. Applicable scientific path: `{status}`.\n\n"+
        'R factorial contrasts are initially single-seed with equal total training budgets and independently DEV-selected checkpoints; selected steps may differ. Only Rstar is repeated. G pair evidence, when activated, keeps both seed signs and does not hide AUX_ONLY or simple VIEW controls. No claim of three universally effective innovations.\n\n'+
        'Purity uses GT only after predictor lock; coefficient×value-norm is a proxy, not a causal decomposition of the normalized representation. Sensitivity to wrong correspondence does not itself establish superiority. H2 assesses new-family supervised-proposal recognition; real SAM2 proposals are post-lock DEV diagnostics. Limited novel support is disclosed without replacing families.\n\n'+
        f"[Claims and signed pairs]({base}claims.json).\n")
    command=(f"/home/ww/miniconda3/envs/ovimap-map/bin/python -u scripts/evaluation/run_ovimap_preservation_routing.py "
             f"--spec configs/static_ovmap/preservation_routing_v1.json --parent-root {binding['lr_parent_root']} "
             f"--output-root {root} --gpu {binding['gpu']} --phase all --resume")
    (docs/'PRESERVATION_ROUTING_HANDOFF.md').write_text(
        '# Preservation routing handoff\n\n'+
        f"Branch: `{binding['branch']}`. Base: `{binding['base_commit']}`. Parent: `{binding['lr_parent_root']}` (read-only). External capacity output: `{root}`.\n\n"+
        'Resolved reproduction command (actual observed invocation is added after subprocess completion):\n\n```bash\n'+command+'\n```\n\n'+
        'Separate controller/FC/SAM2 environments are retained. GPU0 uses the inherited resource lock; NVML query has a driver/library mismatch, while real CUDA is verified by engineering. No global upgrades.\n\n'+
        f"[Selected inference registry]({base}export/weights_registry.json) and [actual strict-load/sample-score roundtrip]({base}export/roundtrip.json). Use `load_exported(registry_file, method, device, FC_identity=..., text_identity=...)`; it loads relative small-weight dependencies and validates exact frozen FC/text identities. Initial MA references and Rstar bases are explicit and deduplicated. FC/text/foundation weights remain external.\n\n"+
        f"[Input/path/hash binding]({base}source_binding.json), [H2 prelock]({base}H2_plan.json), [component curves]({base}training/), [parent audit]({base}parent_audit/receipt.json). Resume stores exact optimizer, LR step, sampler position and Python/NumPy/Torch/CUDA RNG; selected/last optimizer checkpoints remain external. Replayed/discarded work is separately recorded; unknown historical failure time is null.\n\n"+
        'Published artifacts exclude raw licensed scans, GT maps, original/predicted image masks, huge grids and foundation weights. Compact class decisions/embeddings and small heads are included. External publication/final.json is written only after normal push and full local/remote SHA equality; this commit does not claim its own future SHA.\n')
    return result
