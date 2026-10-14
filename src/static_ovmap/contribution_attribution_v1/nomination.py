"""Pure finite evidence ranking and names-only future confirmation locks."""
from pathlib import Path
from functools import cmp_to_key
import hashlib
import re
import numpy as np
from .common import (ConsumptionIndex,verified,read_document,write,immutable_record,unit_identity,cached_record,DependencyError,family)
from .binding import load_binding
from .records import Records,delta

ORDER=('E_GROUP','S','R','E_SOURCE')

def shortlist(candidates):
    active=[r for r in candidates if r['primary_positive_cells']>0 or r['module']=='S' and r['specific_positive_cells']>0]
    def compare(a,b):
        for field in ('specific_positive_cells','primary_positive_cells','lfo_positive_fraction','mean_primary_delta_apall'):
            difference=a[field]-b[field]
            if abs(difference)>1e-10:return -1 if difference>0 else 1
        return ORDER.index(a['module'])-ORDER.index(b['module'])
    active.sort(key=cmp_to_key(compare))
    return [dict(r,status='SHORTLIST_FOR_CONFIRMATION' if not r.get('missing') else 'EVIDENCE_INCOMPLETE',recognition_limited=r['primary_positive_cells']==0) for r in active[:2]]

def candidates(root):
    root=Path(root);binding=load_binding(root);index=ConsumptionIndex();records=Records(root,binding,index)
    stability=verified(root/'stability.json');surface=verified(root/'surface.json');weight=verified(root/'weight_lock.json');tol=1e-10;out=[]
    for module in ('E_SOURCE','E_GROUP','R','S'):
        pairs=[('FG0_FC8','FG2_SURFACE'),('FG0_FC8','S29_FG2_SURFACE')] if module=='S' else [dict(E_SOURCE=('A0','A1'),E_GROUP=('E6','E7'),R=('A1','A3'))[module]]
        primary=[];missing=[];specific=[];lfo=[];cohort_means=[]
        for cohort in binding['cohorts']:
            ds=[]
            for pair in pairs:
                try:
                    d=delta(records.pool(cohort,pair[1])['metrics'],records.pool(cohort,pair[0])['metrics']);primary.append(dict(cohort=cohort,pair=pair,delta=d));ds.append(d['apall'])
                    row=next((r for r in stability['lfo'] if r['cohort']==cohort and tuple(r['pair'])==pair),None)
                    if row and row['stability']['apall']['positive_fraction'] is not None:lfo.append(row['stability']['apall']['positive_fraction'])
                    else:missing.append('LFO '+cohort+' '+str(pair))
                except (DependencyError,FileNotFoundError) as exc:missing.append(str(exc))
            finite=[v for v in ds if v is not None and np.isfinite(v)]
            if finite:cohort_means.append(float(np.mean(finite)))
            if module in ('E_GROUP','R'):
                try:
                    control='E8' if module=='E_GROUP' and weight['weight'] is not None else 'E6' if module=='E_GROUP' else 'U2'
                    method='E7' if module=='E_GROUP' else 'A3';d=delta(records.pool(cohort,method)['metrics'],records.pool(cohort,control)['metrics'])
                    specific.append(dict(cohort=cohort,control=control,method=method,metric='apall',delta=d['apall'],comparability='SAME_FIXED_SOURCE_FUSION' if module=='E_GROUP' else 'FIXED_SURFACE_AND_E;DIFFERENT_RECOVERY_OBSERVATION_LIBRARY_NOT_PURE_SUBSTITUTION'))
                except (DependencyError,FileNotFoundError) as exc:missing.append(str(exc))
        if module=='S':
            matched=surface['history']['matched_step']
            for seed in (17,29):
                a=next((r for r in matched if r['seed']==seed and r['method']=='FG1_VIEW' and r['status']=='COMPLETE'),None);b=next((r for r in matched if r['seed']==seed and r['method']=='FG2_SURFACE' and r['status']=='COMPLETE'),None)
                if a and b:specific.append(dict(seed=seed,step=1000,metric='DEV_A_NOT_MAP_AP',delta=b['metrics']['A']-a['metrics']['A'],comparability='MATCHED_STEP_INITIALIZATION_DRAW_SCHEDULE_FULL200'))
                else:missing.append('MISSING_STEP1000_RECEIPT '+str(seed))
        out.append(dict(module=module,primary=primary,specific=specific,specific_positive_cells=sum(r['delta'] is not None and r['delta']>tol for r in specific) if specific else -1 if module!='E_SOURCE' else 0,
                        primary_positive_cells=sum(r['delta']['apall'] is not None and r['delta']['apall']>tol for r in primary),lfo_positive_fraction=float(np.mean(lfo)) if lfo else -1,
                        mean_primary_delta_apall=float(np.mean(cohort_means)) if cohort_means else -1,missing=missing))
    return out

def prelock(root):
    root=Path(root);path=root/'shortlist_prelock.json';values=candidates(root);selection=shortlist(values)
    return immutable_record(path,dict(status='COMPLETE',selection=selection,candidates=values,rule='SPEC_13_2_FIXED_FINITE_ORDER',selection_depends_on_visuals=False,confirmation_execution='NOT_RUN_BY_CONTRACT'))

def _confirmation(root,binding,index):
    lr=Path(binding['lr_parent_root']);upstream=lr/'external/upstream_sources';manifest=read_document(upstream/'manifest.json',index,sealed=False)
    entries=[r for r in manifest['files'] if r['source_path']=='Tasks/Benchmark/scannetv2_train.txt' and r['repository']=='ScanNet']
    if len(entries)!=1:raise DependencyError('Missing exact official ScanNet train queue entry in LR upstream manifest')
    queue=upstream/entries[0]['local_path']
    roots=[Path(binding[k]) for k in ('fg_parent_root','lr_parent_root','wave2_root','pr_parent_root','observation_root','prior_data_root') if k in binding]
    excluded={family(s) for role in verified(Path(root)/'role_lock.json')['roles'].values() for s in (role if isinstance(role,list) else []) if s.startswith('scene')}
    excluded.update(family(s) for v in binding['cohorts'].values() for s in v);excluded.update(family(s) for s in binding['task_specification']['fusion_dev'])
    paths=[]
    for parent in roots:
        for name in ('source_binding.json','split_manifest.json','H2_plan.json','H2_pretraining_inventory.json','acquisition_lock.json','data_inventory.json','holdout2/receipt.json','data_roles.json','selection_lock.json','role_lock.json'):
            if (parent/name).exists():paths.append(parent/name)
    # The inherited Q source binding directly identifies its old role lists.
    qroot=Path('/mnt/shared/ww/ovimap-module-validation-v1')
    for p in (qroot/'attempt_011/source_binding.json',qroot/'attempt_011/split_manifest.json',qroot/'scannet_runtime_v1/data_inventory.json'):
        if p.exists():paths.append(p)
    exposure=[]
    for path in list(dict.fromkeys(paths)):
        value=read_document(path,index,sealed=False);names=sorted(set(re.findall(r'scene\d{4}(?:_\d{2})?',str(value))))
        excluded.update(family(s) for s in names);exposure.append(dict(source=str(path),file=index.identity(path),names=names))
    missing=[]
    if not queue.is_file():missing.append(str(queue))
    if not exposure:missing.append('FINITE_PARENT_EXPOSURE_RECORDS')
    if missing:return dict(status='PARTIAL_DEPENDENCY',selected=None,missing=missing,confirmation_execution='NOT_RUN_BY_CONTRACT',annotations_parsed=0)
    identity=index.identity(queue)
    if identity['sha256']!=entries[0]['sha256'] or identity['bytes']!=entries[0]['size']:raise ValueError('Official queue bytes differ from inherited manifest')
    families={}
    for scan in queue.read_text().splitlines():
        if not re.fullmatch(r'scene\d{4}_\d{2}',scan.strip()):continue
        scan=scan.strip();f=family(scan)
        if f not in excluded:families.setdefault(f,[]).append(scan)
    ranked=sorted(families,key=lambda f:(hashlib.sha256(('OVMAP-CONFIRM-20261014|'+f).encode()).hexdigest(),f));selected=[min(families[f],key=lambda s:int(s.rsplit('_',1)[1])) for f in ranked[:12]]
    return dict(status='COMPLETE',selected=selected,selected_families=[family(s) for s in selected],requested_families=12,actual_families=len(selected),queue=identity,
                exposure_sources=exposure,excluded_families=sorted(excluded),eligible_family_count=len(families),shortfall=max(0,12-len(selected)),
                independence_claim='NOT_USED_IN_CURRENT_ROUND;FULL_HISTORICAL_NONUSE_NOT_PROVEN',confirmation_execution='NOT_RUN_BY_CONTRACT',annotations_parsed=0,predictions_inspected=0,new_downloads=0)

def run(root):
    root=Path(root);binding=load_binding(root);index=ConsumptionIndex();locked=prelock(root)
    confirmation=_confirmation(root,binding,index)
    from .common import canonical_digest
    key=unit_identity('nominate',dict(shortlist=locked['identity'],confirmation=canonical_digest(confirmation)),producers=('common','records','nomination'))
    previous=cached_record(root/'nominate.json',key)
    if previous:return previous
    selected=write(root/'shortlist.json',dict(status='COMPLETE',selection=locked['selection'],candidates=locked['candidates'],prelock=locked['identity'],selection_depends_on_visuals=False))
    surface=verified(root/'surface.json');registry=read_document(Path(binding['fg_parent_root'])/'export/weights_registry.json',index)
    from .resume import check_frozen_models
    model_lock=check_frozen_models(root)
    inventory=verified(root/'inventory.json');source_locks=[dict(scene=r['scene'],role=r['role'],sources=r['sources'],temperatures=r['temperatures'],request_view_scope=r['request_view_scope']) for r in inventory['role_scenes']]
    plan=write(root/'confirmation_plan.json',dict(status=confirmation['status'],confirmation_execution='NOT_RUN_BY_CONTRACT',scene_nomination=confirmation,
               shortlisted_modules=[r['module'] for r in selected['selection']],locks=dict(weight=verified(root/'weight_lock.json'),FG_heads=registry,verified_weight_and_text_files=model_lock,source_configuration_locks=source_locks,selected_embedding_parity=surface['selected_parity'],role=verified(root/'role_lock.json')['identity'],spec= binding['task_spec_sha256'],vocabularies='ORIGINAL_REPLICA51_SCANNET200_ORDER_AND_FROZEN_TEXT_BYTES',backbone='FROZEN_FC_IDENTITY_IN_SELECTED_REGISTRY'),
               primary_metrics=['OFFICIAL_POOLED_APall','mIoU'],controls=dict(E_GROUP='E6_AND_LOCKED_E8',R='A1_A3_A4_U2_WITH_LIBRARY_AND_BUDGET_DISCLOSURE',S='FG0_FC8_AND_MATCHED_STEP1000_FG1_FG2'),
               required_future_evidence=['INDEPENDENT_FAMILY_CONFIRMATION_WITH_PRELOCKED_CONFIGS','REAL_PROPOSAL_MASKS_NOT_GT_MASK_SUBSTITUTES','HELDOUT_40_CLASS_FULL200_COMPETITION','SAME_HARDWARE_COMPLETE_INPUT_AND_VIEW_BUDGET_COST','CAMFUSION_OR_EQUIVALENT_MATCHED_LEARNED_FUSION_CONTROL'],
               resource_plan=dict(single_GPU='INHERITED_A40',CPU_workers=3,cache_GiB=32,feature_budget='PREDECLARE_BEFORE_FUTURE_ACQUISITION',training='NO_NEW_TRAINING_THIS_ROUND'),deployment='N0_UNCHANGED',submission_evidence_complete=False))
    index.write_memo(root/'nomination/verifications.json')
    return write(root/'nominate.json',dict(status='COMPLETE' if confirmation['status']=='COMPLETE' else 'PARTIAL_DEPENDENCY',input_identity=key,shortlist=selected['identity'],confirmation_plan=plan['identity'],confirmation_execution='NOT_RUN_BY_CONTRACT',submission_evidence_complete=False))
