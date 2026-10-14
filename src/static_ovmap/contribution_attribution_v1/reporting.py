"""Compact recomputation evidence, original-precision tables and bounded claims."""
from pathlib import Path
import csv
import gzip
import json
import shutil
import numpy as np
from .common import (REPO,PHASES,METRICS,FG_BASE,ConsumptionIndex,verified,read_document,write,digest_file,plain,source_digest,family,canonical_digest,DependencyError)
from .binding import load_binding
from .records import Records,delta
from .publication import check_sizes

ARTIFACTS=REPO/'artifacts/static_ovmap/contribution_attribution_v1'
DOCS=REPO/'docs/paper/static_ovmap/contribution_attribution_v1'

def _state_ap(state):
    """The pinned released scorer's terminal PR integration, unchanged."""
    if not state['has_gt']:return None
    if not state['has_pred']:return 0.0
    truth=np.asarray(state['y_true']);score=np.asarray(state['y_score'])
    order=np.argsort(score);sorted_score=score[order];cumulative=np.cumsum(truth[order])
    _,indices=np.unique(sorted_score,return_index=True);n=len(sorted_score);total=cumulative[-1] if len(cumulative) else 0
    cumulative=np.append(cumulative,0);precision=np.zeros(len(indices)+1);recall=np.zeros(len(indices)+1)
    for position,index in enumerate(indices):
        previous=cumulative[index-1];tp=total-previous;fp=n-index-tp;fn=previous+state['hard_false_negatives']
        precision[position]=float(tp)/(tp+fp);recall[position]=float(tp)/(tp+fn)
    precision[-1]=1.;recall[-1]=0.
    return float(np.dot(precision,np.convolve(np.append(np.append(recall[0],recall),0.),[-.5,0,.5],'valid')))

def _gzip(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as compressed:compressed.write((json.dumps(plain(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    temp.replace(path)

def _table(name,rows,columns):
    folder=ARTIFACTS/'tables';folder.mkdir(parents=True,exist_ok=True)
    write(folder/(name+'.json'),dict(status='COMPLETE',rows=rows,columns=columns,metric_unit='FRACTION;DISPLAY_PERCENT'))
    def display(r,k):
        value=r.get(k)
        if isinstance(value,float):return f'{value*100:.2f}' if k in (*METRICS,'A','M','C') else f'{value:.6f}'
        return '—' if value is None else str(value)
    with (folder/(name+'.csv')).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(display(r,k).replace('|','/') for k in columns)+' |' for r in rows]
    (folder/(name+'.md')).write_text('\n'.join(lines)+'\n')
    def tex(value):return value.replace('_','\\_').replace('%','\\%').replace('—','--')
    latex=['\\begin{tabular}{'+'l'*len(columns)+'}','\\toprule',' & '.join(tex(k) for k in columns)+' \\\\','\\midrule']
    latex+=[' & '.join(tex(display(r,k)) for k in columns)+' \\\\' for r in rows];latex+=['\\bottomrule','\\end{tabular}']
    (folder/(name+'.tex')).write_text('\n'.join(latex)+'\n')
    return '\n'.join(lines)

def _compact(root,records,stage_records):
    key=canonical_digest(dict(main=verified(root/'main_score.json')['identity'],inventory=records.inventory['identity'],weight=verified(root/'weight_lock.json')['identity'],report_producer=digest_file(__file__)))
    cache=root/'reporting/compact_receipt.json'
    if cache.exists():
        old=verified(cache)
        if old['input_identity']==key:
            for file in old['output_files']:records.index.identity(REPO/file['path'],file)
            return old['scene_index']
    index=[];external=[];pr_parity=[]
    for cohort,scenes in records.binding['cohorts'].items():
        for scene in scenes:
            methods=list(records.binding['task_specification']['new_main_methods']);names=[m['id'] for m in methods if m['id']!='E8' or verified(root/'weight_lock.json')['weight'] is not None]
            names += ['A0','A1','A2','A3','A4','A5','U2','G3','FG0_FC8','FG1_VIEW','FG2_SURFACE','S29_FG2_SURFACE','FG3_AUX']
            rows=[];seen={};score_bank=[]
            for method in names:
                try:row=records.row(scene,method);score=records.score(scene,method)
                except Exception as exc:
                    if isinstance(exc,DependencyError) and method in ('U2','G3') and cohort=='scannet_cf18':
                        rows.append(dict(method=method,status='MISSING',score_identity=None,metrics=dict.fromkeys(METRICS),reason='PARENT_CONTROL_ONLY_HAS_REPLICA_SCOPE'));continue
                    raise
                identity=score['identity'];rows.append(dict(method=method,score_identity=identity,metrics=row['metrics'],rank_mode=row['rank_mode'],prediction_identity=row['prediction_identity']))
                if identity in seen:continue
                trace=read_document(Path(score['manifest']).with_name('trace.json.gz'),records.index,sealed=False);namespace=records.released(score)
                shape=(len(namespace['dist_threshes']),len(namespace['CLASS_LABELS']),len(namespace['overlaps']));ap=np.full(shape,np.nan)
                compact_states=[]
                for state in trace['states']:
                    reconstructed=_state_ap(state)
                    if (reconstructed is None)!=(state['ap'] is None) or reconstructed is not None and abs(reconstructed-state['ap'])>1e-12:raise ValueError('PR sufficient statistics fail original per-class AP reconstruction')
                    state_key=tuple(state[k] for k in ('distance_index','class_index','overlap_index'));ap[state_key]=np.nan if reconstructed is None else reconstructed
                    compact_states.append({k:state[k] for k in ('distance_index','class_index','class_label','overlap_index','overlap_threshold','y_true','y_score','hard_false_negatives','has_gt','has_pred','ap')})
                values=plain(namespace['compute_averages'](ap))
                for m,k in [('apall','all_ap'),('ap50','all_ap_50%'),('ap25','all_ap_25%')]:
                    if (values[k] is None)!=(score['metrics'][m] is None) or values[k] is not None and abs(values[k]-score['metrics'][m])>1e-12:raise ValueError('Compact terminal PR AP parity differs')
                pr_parity.append(dict(scene=scene,score_identity=identity,all_AP_parity=True))
                score_bank.append(dict(score_identity=identity,metrics=score['metrics'],confusion=score['confusion'],view=score['view'],context=score['context'],PR_sufficient_statistics=compact_states))
                matches=Path(score['manifest']).with_name('matches.json.gz');external.append(dict(scene=scene,score_identity=identity,kind='FULL_MATCHES_REQUIRED_FOR_BOOTSTRAP',file=records.index.identity(matches),regeneration='Rerun task CLI with identical original inputs and a fresh output root; scorer matches retain the released rules.'))
                seen[identity]=True
            readouts=verified(root/'main_predict'/scene/'readouts.json');sources={n:read_document(readouts['source_descriptors'][n]['path'],records.index) for n in ('N','Q','F')}
            recovery_source=read_document(Path(records.binding['area_root'])/'regions'/scene/'G1_FC.json',records.index)
            payload=dict(scene=scene,cohort=cohort,rows=rows,scores=score_bank,valid_ids=readouts['valid_ids'],temperatures=readouts['temperatures'],source_availability=readouts['source_availability'],
                         source_scores={n:dict(model_identity=s.get('model_identity'),text_identity=s.get('text_identity'),valid_ids=s['valid_ids'],objects=s['objects']) for n,s in sources.items()},
                         primary_G1_recovery_F_source={k:recovery_source.get(k) for k in ('identity','model_identity','text_identity','request_manifest_identity','registry_identity','score_kind','valid_ids','objects')},
                         computed_readouts=readouts['methods'],GT_raw_arrays_included=False,raw_masks_included=False)
            path=ARTIFACTS/'scene_evidence'/(scene+'.json.gz');_gzip(path,payload);index.append(dict(scene=scene,path=str(path.relative_to(REPO)),bytes=path.stat().st_size,sha256=digest_file(path)))
            print('COMPACT_PR_VERIFIED',scene,len(score_bank),flush=True)
            records.cache.clear()
    write(ARTIFACTS/'external_matches.json',dict(status='COMPLETE',files=external,not_uploaded=True,reproduction_requires_parent_geometry_GT_inputs=True))
    write(ARTIFACTS/'PR_parity.json',dict(status='COMPLETE',checked=pr_parity,absolute_tolerance=1e-12,source='ORIGINAL_UNCHANGED_TRACE_TERMINAL_STATES'))
    outputs=[*index,*[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest_file(p)) for p in (ARTIFACTS/'external_matches.json',ARTIFACTS/'PR_parity.json')]]
    write(cache,dict(status='COMPLETE',input_identity=key,scene_index=index,output_files=outputs))
    return index

def _claims(root,records):
    surface=verified(root/'surface.json');shortlist=verified(root/'shortlist.json');ledger=[]
    for candidate in shortlist['candidates']:
        module=candidate['module'];signs=[v for r in candidate['primary'] for v in r['delta'].values() if v is not None]
        positive=any(v>1e-10 for v in signs);negative=any(v< -1e-10 for v in signs)
        effect='MIXED' if positive and negative else 'POSITIVE_OBSERVED' if positive else 'NO_POSITIVE_OBSERVED' if signs else 'UNASSESSED'
        attribution=dict(E_SOURCE='PACKAGE_ONLY',E_GROUP='MATCHED_CONTRAST',R='CONDITIONALLY_MATCHED',S='MATCHED_CONTRAST')[module]
        novelty=dict(E_SOURCE='UNRESOLVED',E_GROUP='KNOWN_EQUIVALENT',R='UNRESOLVED',S='CONCRETE_DIFFERENCE_IDENTIFIED')[module]
        ledger.append(dict(module=module,effectiveness=effect,attribution=attribution,novelty=novelty,primary_cells=candidate['primary'],specific_cells=candidate['specific'],shortlisted=any(r['module']==module for r in shortlist['selection']),
                           boundary=dict(E_SOURCE='Heterogeneous evidence increment; no Q_GAIN versus random query control.',E_GROUP='Complete N/Q/F D2 is ordinary 1/4,1/4,1/2 weighting; missing-source and empty-source rules differ.',R='Fixed-surface residual support; recovered coverage is not equivalent to official true positives.',S='Matched-step view versus surface grouping supports structure in DEV; selected map increment remains separate; OOD shuffle is dependence evidence only.')[module],
                           external_sources=dict(E_SOURCE=['L01','L02','L05'],E_GROUP=['L01','L05'],R=['L01','L03'],S=['L02','L03','L04','L06'])[module],independent_confirmation='NOT_RUN_BY_CONTRACT',submission_evidence_complete=False))
    return write(root/'claims_ledger.json',dict(status='COMPLETE',modules=ledger,same_step_recognition=surface['history']['matched_step'],scope='EXPOSED_EXPLORATORY_NOT_CONFIRMATORY',core_or_CVPR_proven_claim=False))

def _partial(root,binding,observed,stages,missing,records):
    """Publish available negative/blocked evidence without inventing absent cells."""
    from .common import DependencyError
    errors=[verified(p) for p in sorted((root/'errors').glob('*.json')) if p.stem in missing]
    execution='FAILED_ENGINEERING' if any(e['status']=='FAILED_ENGINEERING' for e in errors) else 'PARTIAL_DEPENDENCY'
    status=dict(execution_status=execution,evidence_status='ATTRIBUTION_PARTIAL',publication_status='NOT_ATTEMPTED_SNAPSHOT',submission_evidence_complete=False,missing=missing)
    for phase,value in stages.items():
        if phase not in ('main_predict','fusion_dev_predict'):_gzip(ARTIFACTS/'records'/(phase+'.json.gz'),value)
    for name in ('weight_lock','weight_selection','gain_attribution','recovery','lineage','surface','stability','rank','literature','focused_tests','stage_exits','execution_observed'):
        if (root/(name+'.json')).exists():_gzip(ARTIFACTS/'records'/(name+'.json.gz'),verified(root/(name+'.json')))
    write(ARTIFACTS/'partial_status.json',dict(status=execution,**status,actual_errors=errors,available_phases=list(stages)))
    rows=[]
    for cohort in binding['cohorts']:
        for method in ['A0','A1','A2','A3','A4','A5','U2','G3','FG0_FC8','FG2_SURFACE','S29_FG2_SURFACE',*[m['id'] for m in binding['task_specification']['new_main_methods']]]:
            try:pool=records.pool(cohort,method);metrics=pool['metrics'];kind='NEW_CACHED_ANALYSIS' if method.startswith(('E','J')) else 'REUSED_EXACT'
            except (DependencyError,FileNotFoundError,KeyError):metrics=dict.fromkeys(METRICS);kind='MISSING'
            rows.append(dict(cohort=cohort,method=method,**metrics,measurement_kind=kind))
    table=_table('main_system',rows,['cohort','method',*METRICS,'measurement_kind'])
    claim=dict(status='PARTIAL_DEPENDENCY',modules=[dict(module=m,effectiveness='UNASSESSED',attribution='UNRESOLVED',novelty='UNRESOLVED') for m in ('E_SOURCE','E_GROUP','R','S')],submission_evidence_complete=False)
    write(ARTIFACTS/'claims_ledger.json',claim)
    primary=verified(root/'primary_review.json') if (root/'primary_review.json').exists() else dict(status='PENDING_PRIMARY_REVIEW')
    (DOCS/'RESULTS.md').write_text('# 贡献归因：部分执行结果\n\n'+json.dumps(status,ensure_ascii=False)+'\n\n'+table+'\n\n缺失单元为null；可用负结果照常保留，未完成的归因不作有效或新颖性结论。\n')
    (DOCS/'CLAIMS_LEDGER.md').write_text('# Claims\n\n'+json.dumps(claim,ensure_ascii=False,indent=2)+'\n')
    plan=verified(root/'confirmation_plan.json') if (root/'confirmation_plan.json').exists() else dict(status='PARTIAL_DEPENDENCY',selected=None,missing=['COMPLETED_NOMINATION_INPUTS'],confirmation_execution='NOT_RUN_BY_CONTRACT')
    (DOCS/'CONFIRMATION_PLAN.md').write_text('# Confirmation — NOT_RUN_BY_CONTRACT\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (DOCS/'REVIEW.md').write_text('# Primary review\n\n'+json.dumps(primary,ensure_ascii=False,indent=2)+'\n')
    (DOCS/'HANDOFF.md').write_text('# Partial execution handoff\n\n'+json.dumps(dict(**status,REPO=str(REPO),OUT=str(root),branch=binding['task_branch'],observed_scientific_exit=observed,errors=errors),ensure_ascii=False,indent=2)+'\n\n保留原完整单元；修复只针对缺失阶段及受影响下游。没有新增训练、RGB、SAM或建图权限。\n')
    files=sorted(p for p in ARTIFACTS.rglob('*') if p.is_file() and p.name!='artifact_index.json');check_sizes(files)
    artifact=write(ARTIFACTS/'artifact_index.json',dict(status=execution,files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest_file(p)) for p in files],numeric_source_digest=source_digest(),report_source_sha256=digest_file(__file__),spec_sha256=binding['task_spec_sha256'],raw_inputs_uploaded=False))
    return write(root/'report.json',dict(status=execution,**status,artifact_index=artifact['identity'],primary_review=primary['status']))

def _interpretation(root,records,stages):
    lines=[]
    def pp(value):return 'undefined' if value is None else f'{value*100:+.4f}pp'
    for cohort in records.binding['cohorts']:
        def change(before,after):return delta(records.pool(cohort,after)['metrics'],records.pool(cohort,before)['metrics'])
        e=change('A0','A1');r=change('A1','A3');g=change('E6','E7')
        lines.append(f'{cohort}: E_SOURCE APall {pp(e["apall"])}, mIoU {pp(e["miou"])}；R|E APall {pp(r["apall"])}, mIoU {pp(r["miou"])}；E_GROUP(D2−equal) APall {pp(g["apall"])}, mIoU {pp(g["miou"])}。')
        if verified(root/'weight_lock.json')['weight'] is not None:
            d=change('E8','E7');lines.append(f'{cohort}: D2−锁定普通融合 E8 APall {pp(d["apall"])}, mIoU {pp(d["miou"])}；E8仅由四个独立融合DEV场景选择，未在REG上改权重。')
        for seed,method in [(17,'FG2_SURFACE'),(29,'S29_FG2_SURFACE')]:
            d=change('FG0_FC8',method);lines.append(f'{cohort}: selected FG2 seed{seed}−FC8 整图 APall {pp(d["apall"])}, mIoU {pp(d["miou"])}；此行与同step1000识别对照分开。')
        recovery=verified(root/'recovery.json');rows=[r for r in recovery['scene_rows'] if r['cohort']==cohort]
        events=[e for r in rows for e in r['official_changes'] if e['threshold']==.5]
        lines.append(f'{cohort}: 新恢复实例{sum(r["new_instances"] for r in rows)}，投影≥100点{sum(r["new_projected_evaluable_owners"] for r in rows)}；IoU50新增definite TP score entries {sum(e["new_definite_TP_score_entries"] for e in events)}、FP entries {sum(e["new_definite_FP_score_entries"] for e in events)}。覆盖、唯一几何匹配、官方score-entry分别统计；事件数不是互斥对象数。')
    for seed in (17,29):
        matched=stages['surface']['history']['matched_step'];a=next(r for r in matched if r['seed']==seed and r['method']=='FG1_VIEW');b=next(r for r in matched if r['seed']==seed and r['method']=='FG2_SURFACE')
        if a['status']==b['status']=='COMPLETE':lines.append(f'S seed{seed}同1000step DEV A：FG1 {a["metrics"]["A"]*100:.4f}% → FG2 {b["metrics"]["A"]*100:.4f}%（{pp(b["metrics"]["A"]-a["metrics"]["A"])}），初始化/抽样记录匹配，全200类竞争。')
    for pair,cohorts in stages['rank']['pools'].items():
        for cohort,value in cohorts.items():
            e=value['effects']['apall']
            lines.append(f'{cohort} {pair} rank诊断：固定旧rank改类别 {pp(e["E_add"])}；固定旧类别改rank {pp(e["R_add"])}；交互 {pp(e["interaction"])}。非对角仅诊断，不能加入正式主表或解释为训练因果份额。')
    for b in stages['stability']['bootstrap']:
        ci=b['intervals']['apall'];lines.append(f'{b["cohort"]} {"→".join(b["pair"])}：2000家族配对draw的ΔAPall 95%分位区间 [{pp(ci["lower_025"])}, {pp(ci["upper_975"])}]；defined={ci["defined_draws"]}，undefined={ci["undefined_draws"]}。已曝光探索性区间，非独立确认。')
    return lines

def _historical_context(root,binding,index):
    rows=[];missing=[]
    for parent,path,methods,scope in [
        (binding['minimal_root'],'result_store.json',('IR02_NEAREST_ATTACH','IR03_EVIDENCE_ATTACH'),'DIFFERENT_RECOVERY_OPERATOR_LIBRARY_NOT_PURE_R_SUBSTITUTION'),
        (binding['fg_parent_root'],'evaluation/store.json',('FG4_ROUTE','S29_FG4_ROUTE'),'HISTORICAL_CONTROLLER_CONTEXT_ONLY_NO_NEW_FORWARD')]:
        target=Path(parent)/path
        if not target.exists():missing.append(str(target));continue
        store=read_document(target,index)
        for cohort,bank in store['pooled_metrics'].items():
            for method in methods:
                if method not in bank:missing.append(cohort+' '+method);continue
                pooled=bank[method]
                rows.append(dict(method=method,cohort=cohort,metrics=pooled['metrics'],pool_identity=pooled['identity'],source_root=parent,measurement_kind='REUSED_EXACT',comparability=scope,exposure='EXPOSED_HISTORICAL'))
    w2=Path(binding['wave2_root']);wave={}
    for name in ('final_result.json','selection.json','selection_lock.json'):
        path=w2/name
        if path.exists():wave[name]=dict(file=index.identity(path),record=read_document(path,index,sealed=name!='final_result.json'))
        else:missing.append(str(path))
    value=write(ARTIFACTS/'historical_context.json',dict(status='COMPLETE' if not missing else 'PARTIAL_DEPENDENCY',rows=rows,Wave2=wave,missing=missing,new_model_forwards=0,new_scene_assign_calls=0,not_current_method_or_budget=True))
    _table('historical_controls',[dict(r,**r['metrics']) for r in rows],['cohort','method',*METRICS,'comparability']);return value

def _recognition_records(binding,surface,index):
    parent=Path(binding['fg_parent_root']);reference=read_document(parent/'recognition/dev_FC8.json',index);units={};aliases=[]
    for kind,rows in [('MATCHED_STEP1000',surface['history']['matched_step']),('OWN_SELECTED',surface['history']['selected'])]:
        for row in rows:
            if row.get('status','COMPLETE')!='COMPLETE':continue
            path=parent/f'training/seed{row["seed"]}'/row['method']/f'dev_{row["step"]:04d}.json';source=read_document(path,index)
            if source['identity']!=row['receipt']:raise ValueError('Published S recognition receipt differs')
            units[source['identity']]=dict(file=index.identity(path),records=source['records'],arrays=source['arrays'],full200_competitors=True)
            aliases.append(dict(kind=kind,method=row['method'],seed=row['seed'],step=row['step'],receipt=source['identity']))
    _gzip(ARTIFACTS/'recognition_records.json.gz',dict(status='COMPLETE',reference_FC8=dict(identity=reference['identity'],records=reference['records']),units=units,aliases=aliases,
           original_full_arrays_remain_in_OUT_parent=True,normalized_records_recompute_A_M_C=True,selected_and_matched_step_separate=True))

def run(root):
    root=Path(root);binding=load_binding(root);DOCS.mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(parents=True,exist_ok=True)
    from .runner import OUTPUTS
    index=ConsumptionIndex();records=Records(root,binding,index);observed=verified(root/'execution_observed.json');stages={p:verified(root/OUTPUTS[p]) for p in PHASES if (root/OUTPUTS[p]).exists()}
    missing=[p for p in PHASES if p not in stages or stages[p]['status'] not in ('COMPLETE','ALL_PREDICTIONS_LOCKED','NOT_SELECTED_NO_FUSION_DEV','NOT_RUN_BY_CONTRACT')]
    if observed['exit_code']!=0 or missing:return _partial(root,binding,observed,stages,missing,records)
    execution='COMPLETE';claims=_claims(root,records)
    historical_context=_historical_context(root,binding,index)
    attribute=stages['attribute'];rates=[]
    for row in attribute['source_summaries']:
        rates.append(dict(row,NQ_common_error_fraction=row['NQ_common_errors']/row['objects'] if row['objects'] else None,
                          F_corrects_NQ_fraction=row['F_corrects_NQ']/row['NQ_common_errors'] if row['NQ_common_errors'] else None,
                          fusion_corrected_fraction=row['fusion_corrected']/row['objects'] if row['objects'] else None,
                          fusion_harmed_fraction=row['fusion_harmed']/row['objects'] if row['objects'] else None))
    write(ARTIFACTS/'source_error_rates.json',dict(status='COMPLETE',rows=rates,source_attribute=attribute['identity'],support='SAME_ALL_THREE_AVAILABLE_AND_DESCRIPTIVE_IOU>0.5',F_correction_denominator='NQ_common_errors',zero_denominator=None))
    _table('source_error_rates',rates,['cohort','stratum','objects','NQ_common_errors','F_corrects_NQ','F_corrects_NQ_fraction','fusion_corrected','fusion_harmed'])
    copied=[]
    for name in (*PHASES,'weight_selection','weight_lock','model_input_lock','gain_attribution','lineage','recovery','literature','shortlist','shortlist_prelock','confirmation_plan','claims_ledger','focused_tests','stage_exits','execution_observed','primary_domain_proof'):
        path=root/(name+'.json')
        if not path.is_file():continue
        # Giant prediction manifests are private; their compact rows are below.
        if name in ('main_predict','fusion_dev_predict'):continue
        target=ARTIFACTS/'records'/(name+'.json.gz');_gzip(target,verified(path));copied.append(str(target.relative_to(REPO)))
    for path in (root/'repairs').glob('*.json'):_gzip(ARTIFACTS/'records'/('repair_'+path.stem+'.json.gz'),verified(path))
    for p in (root/'stability').glob('*/*_bootstrap.npz'):
        dest=ARTIFACTS/'statistics'/p.parent.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    if (root/'qualitative').exists():
        for p in (root/'qualitative').rglob('*'):
            if p.is_file() and p.suffix in ('.png','.svg','.pdf','.json'):
                dest=ARTIFACTS/'figures'/p.relative_to(root/'qualitative');dest.parent.mkdir(parents=True,exist_ok=True)
                if p.suffix=='.svg':dest.write_bytes(b'\n'.join(line.rstrip(b' \t\r') for line in p.read_bytes().split(b'\n')))
                else:shutil.copyfile(p,dest)
    compact=_compact(root,records,stages)
    from .pr_audit import run as verify_compact_pools
    compact_pool_proof=verify_compact_pools(root,records,compact,ARTIFACTS)
    mainrows=[]
    for cohort in binding['cohorts']:
        for method in ['A0','A1','A2','A3','A4','A5','U2','G3','FG0_FC8','FG1_VIEW','FG2_SURFACE','S29_FG2_SURFACE','FG3_AUX',*stages['main_score']['methods']]:
            try:
                p=records.pool(cohort,method);mainrows.append(dict(cohort=cohort,method=method,**p['metrics'],measurement_kind='NEW_CACHED_ANALYSIS' if method in stages['main_score']['methods'] else 'REUSED_EXACT'))
            except DependencyError:
                if method not in ('U2','G3') or cohort!='scannet_cf18':raise
                mainrows.append(dict(cohort=cohort,method=method,**dict.fromkeys(METRICS),measurement_kind='MISSING',reason='PARENT_CONTROL_ONLY_HAS_REPLICA_SCOPE'))
    mainrows+=verified(root/'literature.json')['paper_rows'];mainrows=[dict(r,**r.get('metrics',{})) for r in mainrows]
    system_table=_table('main_system',mainrows,['cohort','method',*METRICS,'measurement_kind'])
    ablationrows=[]
    for r in claims['modules']:
        for cell in r['primary_cells']:ablationrows.append(dict(module=r['module'],cohort=cell['cohort'],pair=' -> '.join(cell['pair']),**cell['delta'],attribution=r['attribution']))
    for cohort in binding['cohorts']:
        if verified(root/'weight_lock.json')['weight'] is not None:ablationrows.append(dict(module='E_GROUP_STRONG_CONTROL',cohort=cohort,pair='E8 -> E7',**delta(records.pool(cohort,'E7')['metrics'],records.pool(cohort,'E8')['metrics']),attribution='MATCHED_CONTRAST'))
    for gain in verified(root/'gain_attribution.json')['rows']:
        for term in ('E_add','E_cond','R_add','R_cond','interaction','mean_marginal_E','mean_marginal_R','total'):
            label=term if gain['kind']=='ER' else dict(E_add='S_add_F_ONLY',E_cond='S_add_D2',R_add='FUSION_add_FC8',R_cond='FUSION_add_FG2',interaction='S_FUSION_interaction',mean_marginal_E='mean_marginal_S',mean_marginal_R='mean_marginal_FUSION',total='total')[term]
            ablationrows.append(dict(module=gain['kind']+(f'_S{gain["seed"]}' if gain['seed'] else ''),cohort=gain['cohort'],pair=label,**{m:None if gain['pooled']['effects'][m] is None else gain['pooled']['effects'][m][term] for m in METRICS},attribution='DESCRIPTIVE_FACTORIAL_E_R' if gain['kind']=='ER' else 'DESCRIPTIVE_FACTORIAL_S_FUSION'))
    ablation_table=_table('module_attribution',ablationrows,['module','cohort','pair',*METRICS,'attribution'])
    surface=stages['surface'];costrows=[dict(operation=k,actual=surface['workload'][v],limit=limit,scope='NEW_THIS_TASK') for k,v,limit in [('head_calls','new_head_calls_this_invocation',4000),('RGB_encodings','new_rgb_encodings',0),('optimizer_updates','new_optimizer_updates',0),('SAM_proposals','new_sam_proposals',0),('map_construction','new_map_construction',0)]]
    _recognition_records(binding,surface,index)
    _table('S_matched_step',[dict(r,**r.get('metrics',{})) for r in surface['history']['matched_step']],['seed','step','method','A','M','C','status'])
    _table('S_historical_selected',[dict(r,status='COMPLETE',**r.get('metrics',{})) for r in surface['history']['selected']],['seed','step','method','A','M','C','status'])
    _table('S_selected_and_five_shuffles',[dict(method=name,seed=value['seed'],selected_step=value['selected_step'],intervention=mode,**metrics) for name,value in surface['intervention'].items() for mode,metrics in value['metrics'].items()],['method','seed','selected_step','intervention','A','M','C','net_corrections'])
    _table('DEV_all_15_weights',[dict(r,**r.get('metrics',{})) for r in verified(root/'weight_selection.json')['candidates']],['method','weight',*METRICS])
    costrows += [dict(operation='main_scene_rows',actual=stages['main_score']['logical_scene_rows'],limit=312,scope='LOGICAL_CONFIG_ROWS'),dict(operation='main_pools',actual=stages['main_score']['logical_pools'],limit=24,scope='LOGICAL_ORDERED_POOLS'),dict(operation='rank_off_diagonal_assigns',actual=stages['rank']['physical_scene_assign_calls'],limit=104,scope='DIAGNOSTIC'),dict(operation='bootstrap_draw_pairs',actual=stages['stability']['bootstrap_draw_pairs'],limit=8000,scope='MATCHES_ONLY_EXPOSED_EXPLORATORY'),dict(operation='confirmation_families_evaluated',actual=0,limit=None,scope='NOT_RUN_BY_CONTRACT')]
    costrows += [dict(operation='phi_calls',actual=surface['workload']['total_phi_calls'],limit=None,scope='NEW_FROZEN_FEATURE_REPOOLING'),dict(operation='ordinary_FC_region_readouts',actual=surface['workload']['ordinary_FC_view_readouts'],limit=None,scope='FROM_CACHED_GRIDS_NO_RGB_ENCODING'),dict(operation='confirmation_APall',actual=None,limit=None,scope='NOT_RUN_BY_CONTRACT')]
    costrows=[dict(r,evidence_status='NOT_RUN_BY_CONTRACT' if r['scope']=='NOT_RUN_BY_CONTRACT' else 'TECHNICAL_COST_OF_EXPOSED_ANALYSIS') for r in costrows]
    cost_table=_table('budget_evidence',costrows,['operation','actual','limit','scope','evidence_status'])
    recovery_log=root/'logs/attribute.log';recovery_units=sum(line.startswith('RECOVERY_SCENE ') for line in recovery_log.read_text().splitlines())
    cost=write(ARTIFACTS/'cost_evidence.json',dict(status='COMPLETE',head=surface['workload'],stages={p:{k:v for k,v in stages[p].items() if k.startswith(('physical_','nested_','matches_only_','logical_','engineering_','recovered_','new_assign_')) or k in ('wall_seconds','CPU_workers','BLAS_threads')} for p in ('fusion_dev_score','main_score','attribute','rank','stability')},
               compact_pooled_metric_check=dict(proof=compact_pool_proof['identity'],original_compute_averages_calls=compact_pool_proof['original_compute_averages_calls'],new_scene_assign_calls=0,new_evaluate_matches_calls=0,new_head_calls=0),recovery_geometry_diagnostic_calls=dict(completed_scene_units_including_failed_prefix=recovery_units,method_threshold_calls=recovery_units*2*3,successful_analysis_calls=len(verified(root/'recovery.json')['scene_rows'])*2*3,source_log=dict(path=str(recovery_log),bytes=recovery_log.stat().st_size,sha256=digest_file(recovery_log)),scope='ONE_TO_ONE_A1_A3_THREE_THRESHOLDS;NOT_SCENE_ASSIGN_OR_MODEL_CALLS'),actual_stage_exits=verified(root/'stage_exits.json'),counter_scope='NEW_TASK_AND_EXACT_REUSE_SEPARATE',nested_and_phase_wall_times_not_additive=True,confirmation_execution='NOT_RUN_BY_CONTRACT',confirmation_metrics=None))
    ledger=[];source_headers={}
    for r in mainrows:
        if r['measurement_kind']=='PAPER_REPORTED_NOT_MATCHED':
            fields=dict.fromkeys(('ordered_scenes','family_ids','geometry_identity','owner_partition_digest','semantic_digest','rank_digest','source_model_and_text_ids','source_availability','request_view_scope','supervision','temperatures','recipe','seed_step','scorer_context','pool_identity'))
            ledger.append(dict(r,**fields,experiment_ref='L01 Table 3',source_root='https://arxiv.org/html/2603.26541v1',exposure='PAPER_REPORTED_NOT_MATCHED',protocol_match=False));continue
        if r['measurement_kind']=='MISSING':
            ledger.append(dict(r,experiment_ref=r['method'],source_root=binding['area_root'],ordered_scenes=binding['cohorts'][r['cohort']],family_ids=[family(s) for s in binding['cohorts'][r['cohort']]],geometry_identity=None,owner_partition_digest=None,semantic_digest=None,rank_digest=None,source_model_and_text_ids=None,source_availability=None,request_view_scope=None,supervision=None,temperatures=None,recipe=None,seed_step=None,scorer_context=None,pool_identity=None,exposure='NO_AVAILABLE_PARENT_MEASUREMENT'));continue
        scene_order=binding['cohorts'][r['cohort']];pool=records.pool(r['cohort'],r['method']);scenes=[]
        for scene in scene_order:
            row=records.row(scene,r['method']);score=records.score(scene,r['method']);manifest=read_document(records.manifest(scene,r['method']),index,sealed=False);readout=verified(root/'main_predict'/scene/'readouts.json')
            if scene not in source_headers:
                parsed={name:read_document(descriptor['path'],index) for name,descriptor in readout['source_descriptors'].items()}
                source_headers[scene]={name:{k:value.get(k) for k in ('model_identity','text_identity','request_manifest_identity')} for name,value in parsed.items()}
            source_ids=source_headers[scene]
            scenes.append(dict(scene=scene,geometry_identity=manifest['geometry'],owner_partition_digest=manifest['owner_ids']['sha256'],semantic_digest=manifest['semantic_labels']['sha256'],rank_digest=canonical_digest(manifest['instance_ranks']),scorer_context=score['context'],source_model_and_text_ids=source_ids,original_prediction_manifest=index.identity(records.manifest(scene,r['method'])),source_availability=dict(scene_evidence=str(ARTIFACTS.relative_to(REPO)/'scene_evidence'/(scene+'.json.gz')),field='source_availability'),temperatures=readout['temperatures']))
        domains={field:{s['scene']:s[field] for s in scenes} for field in ('geometry_identity','owner_partition_digest','semantic_digest','rank_digest','scorer_context','source_availability','temperatures')}
        ledger.append(dict(experiment_ref=r['method'],source_root=str(root if r['method'] in stages['main_score']['methods'] else binding['area_root'] if r['method'] in ('A0','A1','A2','A3','A4','A5','U2','G3') else binding['fg_parent_root']),cohort=r['cohort'],ordered_scenes=scene_order,family_ids=[family(s) for s in scene_order],scene_domains=scenes,**domains,
                           source_model_and_text_ids=dict(FC='6132b1b1da69a7a3bcd650586f0bc7e9aea2b64d8183a004b1a61c39892c8b44',per_scene_incumbent_sources={s['scene']:s['source_model_and_text_ids'] for s in scenes},FG_selected_head_reference=f'https://github.com/Orangekostar/oviovo/blob/{FG_BASE}/artifacts/static_ovmap/frozen_fc_routing_v1/export/weights_registry.json'),
                           request_view_scope='ELIGIBLE_F_PREFIX8_WITH_FIXED_NONELIGIBLE_G1_LABELS;D2_ALSO_USES_PARENT_N_Q;FULL_SYSTEM_INPUT_NOT_EQUAL_F_BUDGET' if r['method'].startswith(('FG','S29','J')) else 'ORIGINAL_INCUMBENT_N_Q_F_REQUESTS_AND_METHOD_SPECIFIC_RECOVERY;EXACT_PREDICTION_MANIFESTS_IN_SCENE_DOMAINS',supervision='622_BASE_ORIGINALS_24_FAMILIES_INHERITED_ONLY' if r['method'] in ('FG1_VIEW','FG2_SURFACE','S29_FG2_SURFACE','FG3_AUX','J1_FG2_S17_FONLY','J2_FG2_S29_FONLY') else 'NO_LEARNED_HEAD_SUPERVISION',recipe=r['method'],seed_step=dict(seed=29,step=2000) if r['method'] in ('S29_FG2_SURFACE','J2_FG2_S29_FONLY') else dict(seed=17,step=500) if r['method']=='FG1_VIEW' else dict(seed=17,step=1000) if r['method'] in ('FG2_SURFACE','FG3_AUX','J1_FG2_S17_FONLY') else None,pool_identity=pool['identity'],exposure='EXPOSED_EXPLORATORY_NOT_CONFIRMATORY',measurement_kind=r['measurement_kind']))
    write(ARTIFACTS/'experiment_ledger.json',dict(status='COMPLETE',units=ledger))
    files=sorted(p for p in ARTIFACTS.rglob('*') if p.is_file() and p.name!='artifact_index.json');total=check_sizes(files)
    artifact_index=write(ARTIFACTS/'artifact_index.json',dict(status='COMPLETE',scene_evidence=compact,files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest_file(p)) for p in files],task_artifact_bytes=total,numeric_source_digest=source_digest(),report_source_sha256=digest_file(__file__),spec_sha256=binding['task_spec_sha256'],raw_inputs_uploaded=False))
    status=dict(execution_status=execution,evidence_status='ATTRIBUTION_COMPLETE_EXPOSED_ONLY' if execution=='COMPLETE' else 'ATTRIBUTION_PARTIAL',publication_status='NOT_ATTEMPTED_SNAPSHOT',submission_evidence_complete=False,missing=missing)
    results=['# OVMap 贡献归因实测结果','',json.dumps(status,ensure_ascii=False),'','本轮不新增训练、RGB编码、SAM提案或建图。所有回归场景已曝光；后续确认仅登记，未执行。','',*_interpretation(root,records,stages),'',
             '## 系统与论文原版参照','',system_table,'','外部 OVI-MAP 数字来自论文 Table 3，标记 PAPER_REPORTED_NOT_MATCHED；不以这些差值宣称同协议胜出。本地 voxel 描述0.01m与论文0.1m的差异保留。U2/G3历史对照只有Replica记录，CF18相应辅助格为MISSING/null，不扩展原观测库或新增推理填充。12个正式新配置均完整覆盖26场景。','',
             '## 模块增量（显示为百分点）','',ablation_table,'','E_SOURCE 是增加来源的证据包差值；E_GROUP 为融合规则差值。完整三源 D2 与普通(0.25,0.25,0.5)数学等价；缺源/全缺源另见八种availability表。A5是F＋R；E0纯N不默认等于A0原生标签。','',
             '## 预算与证据范围','',cost_table,'','计时是实际缓存分析；各阶段存在嵌套/重叠，不能相加为冷启动端到端耗时或30FPS。本轮3324次冻结head包括3312次DEV干预和12次TRAIN工程检查；训练样例仅2个，不是重测TRAIN。','',
             '## 结构与对应','',f'同step1000两seed的FG1/FG2记录、各自selected结果和5次shuffle均已保留。新原样组与继承embedding逐位一致：{surface["selected_parity"]}。对应打乱的下降仅支持当前模型依赖，不能推断随机分组重新训练必然失败。','',
             '## 确认优先级','',json.dumps(verified(root/'shortlist.json')['selection'],ensure_ascii=False,indent=2),'','该名单是 SHORTLIST_FOR_CONFIRMATION，不是已证明的CVPR核心贡献。仍需独立家族、真实提案、heldout40类与统一输入/硬件预算，以及同特征学习式融合对照。','',
             '完整四格、逐scene/family/class、错误路径、官方score-entry/多对一/唯一几何三种口径、显式rank交叉、LFO及8000draw-pairs详见相邻机器工件。',
             '',f'[工件索引](../../../../artifacts/static_ovmap/contribution_attribution_v1/artifact_index.json) · [贡献账本](../../../../artifacts/static_ovmap/contribution_attribution_v1/experiment_ledger.json)']
    (DOCS/'RESULTS.md').write_text('\n'.join(results)+'\n')
    (DOCS/'CLAIMS_LEDGER.md').write_text('# Claim ledger\n\n'+json.dumps(claims,ensure_ascii=False,indent=2)+'\n\n具体近邻：[原始来源与区别表](../../../../artifacts/static_ovmap/contribution_attribution_v1/records/literature.json.gz)。\n')
    plan=verified(root/'confirmation_plan.json');(DOCS/'CONFIRMATION_PLAN.md').write_text('# Confirmation plan — NOT_RUN_BY_CONTRACT\n\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    primary=verified(root/'primary_review.json') if (root/'primary_review.json').exists() else dict(status='PENDING_PRIMARY_REVIEW',scope='SPEC_0_TO_16_AND_RENDERED_OUTPUTS')
    (DOCS/'REVIEW.md').write_text('# Primary contract review\n\n'+json.dumps(primary,ensure_ascii=False,indent=2)+'\n')
    handoff=['# OVMap contribution attribution handoff','',f'REPO: {REPO}',f'OUT: {root}',f'Branch: {binding["task_branch"]}',f'Control Python: /home/ww/miniconda3/envs/ovimap-map/bin/python',f'FC Python: {binding["FC_python"]}',f'GPU: {binding["gpu"]}; inherited queued lock /mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{binding["gpu"]}.lock','',
             '```bash',f'cd {REPO}',f'/home/ww/miniconda3/envs/ovimap-map/bin/python -u scripts/evaluation/run_ovmap_contribution_attribution_v1.py --spec configs/static_ovmap/contribution_attribution_v1.json --parent-root {binding["fg_parent_root"]} --output-root {root} --gpu {binding["gpu"]} --phase all --resume','```','',
             f'普通权重锁：{verified(root/"weight_lock.json")["weight"]}；selection_role=FUSION_DEV。',f'完成/缺失状态：{status}',f'科学子进程已观察退出：{observed["exit_code"]}。历史失败见stage_exits，最终成功阶段另列；没有预写父all退出。',
             f'Frozen heads出自FG_BASE {FG_BASE}，没有新模型权重。完整matches仍在OUT；Git PR充分统计可复核AP，bootstrap需要完整matches，不以精简字段冒充可重做几何匹配。',
             '',f'发布收据边界：C1发布后写PUBLICATION_RECEIPT.json；C2和最终all真实退出只写OUT外部。当前报告快照不预写未来SHA。','',
             '表格重建：使用相同binding和冻结输出运行 --phase report；不会调用head或评分。数据角色、直接输入、实际hash和机器结果见artifact_index和experiment_ledger。部署N0_UNCHANGED，confirmation=NOT_RUN_BY_CONTRACT。']
    (DOCS/'HANDOFF.md').write_text('\n'.join(handoff)+'\n')
    index.write_memo(root/'reporting/verifications.json')
    return write(root/'report.json',dict(status='COMPLETE' if execution=='COMPLETE' else 'PARTIAL_DEPENDENCY',**status,artifact_index=artifact_index['identity'],numeric_source_digest=source_digest(),report_source_sha256=digest_file(__file__),primary_review=primary['status']))
