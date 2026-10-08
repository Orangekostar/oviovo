"""Three canonical tables, compact real evidence and ordinary branch publication."""

from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import shutil
import subprocess

import numpy as np

from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,read
from .binding import REPO,seal,free_space

ARTIFACT=Path('artifacts/static_ovmap/source_preserving_update_v1')
DOCS=Path('docs/paper/static_ovmap/source_preserving_update_v1')
REPORT_DOCS=Path('docs/paper/static_ovmap')
METHOD_NAMES={'SU00_D2':'Inherited D2','SU01_G1':'Inherited G1',
    'SU02_HARD_MATCHED':'Hard overwrite (matched)','SU03_STABLE_MATCHED':'Stable overwrite (matched)',
    'SU04_F_REPLACE':'Replace F','SU05_F_BLEND':'Blend F+A','SU06_F_COARSE':'Blend F+C',
    'SU07_GLOBAL_BLEND':'Global same-A-mass blend','SU08_PAIRED_DELTA':'Paired A-C delta'}


def optional(path,default):
    return read(path) if Path(path).is_file() else default


def gz_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    path.write_bytes(gzip.compress(data,mtime=0))


def pack_vectors(value,arrays):
    """Deduplicate exact finite numeric JSON vectors; preserve null and integer types."""
    if isinstance(value,dict):return {k:pack_vectors(v,arrays) for k,v in value.items()}
    if isinstance(value,list):
        if len(value)>8 and all(v is None or type(v) in (int,float) for v in value) and all(
                not isinstance(v,int) or abs(v)<=2**53 for v in value):
            data=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
            key=hashlib.sha256(data).hexdigest()
            arrays.setdefault(key,np.asarray([0 if v is None else v for v in value],np.float64))
            result={'array_reference':key}
            nulls=[i for i,v in enumerate(value) if v is None]
            integers=[i for i,v in enumerate(value) if type(v) is int]
            if nulls:result['null_indices']=nulls
            if integers:result['integer_indices']=integers
            return result
        return [pack_vectors(v,arrays) for v in value]
    return value


def unpack_vectors(value,arrays):
    """Restore compact decisions, checking every referenced original JSON vector hash."""
    if isinstance(value,dict):
        if 'array_reference' in value and set(value)<= {'array_reference','null_indices','integer_indices'}:
            key=value['array_reference'];result=arrays[key].tolist()
            for i in value.get('null_indices',[]):result[i]=None
            for i in value.get('integer_indices',[]):result[i]=int(result[i])
            actual=hashlib.sha256(json.dumps(result,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if actual!=key:raise ValueError('compact vector content differs from original JSON')
            return result
        return {k:unpack_vectors(v,arrays) for k,v in value.items()}
    if isinstance(value,list):return [unpack_vectors(v,arrays) for v in value]
    return value


def display(value):
    if value is None:return '—'
    if isinstance(value,bool):return 'yes' if value else 'no'
    if isinstance(value,float):return f'{value:.3f}'
    return str(value)


def table_files(directory,stem,rows,columns,caption):
    directory.mkdir(parents=True,exist_ok=True)
    atomic_write_json(directory/(stem+'.json'),{'caption':caption,'columns':columns,'rows':rows})
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (directory/(stem+'.csv')).write_text(out.getvalue())
    md=caption+'\n\n| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'
    md+=''.join('| '+' | '.join(display(row.get(k)) for k in columns)+' |\n' for row in rows)
    (directory/(stem+'.md')).write_text(md)
    def tex(v):return display(v).replace('—','--').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    latex_columns=[k for k in columns if k not in ('Rule','Evidence')]
    short={'Replica_apall_pct':'R AP','Replica_ap50_pct':'R AP50','Replica_miou_pct':'R IoU',
        'CF18_apall_pct':'CF AP','CF18_ap50_pct':'CF AP50','CF18_miou_pct':'CF IoU','Target_met':'Pass',
        'Material_met':'Material','Delta_apall_pp':r'$\Delta$AP','Delta_ap50_pp':r'$\Delta$AP50',
        'Delta_miou_pp':r'$\Delta$IoU','GT50_gained':'GT+','GT50_lost':'GT-',
        'Wrong_to_right':'W-R','Right_to_wrong':'R-W','Output_identity_tie':'Tie',
        'Applied_changes':'Changed','Paired_FULL':'Pairs','CPU_ms_scene':'CPU ms'}
    def cell(v):
        if isinstance(v,str):
            for m in METHOD_NAMES:v=v.replace(m,m.split('_')[0]+(' (matched)' if m in ('SU02_HARD_MATCHED','SU03_STABLE_MATCHED') else ''))
            v=v.replace('replica8','R8').replace('scannet_cf18','CF18')
        return tex(v)
    alignment=''.join('l' if k in ('Method','Cohort','Contrast') else 'r' for k in latex_columns)
    content=r'\begin{tabular}{'+alignment+'}\n'+r'\toprule'+'\n'
    content+=' & '.join(short.get(k,tex(k)) for k in latex_columns)+r' \\'+'\n'+r'\midrule'+'\n'
    content+=''.join(' & '.join(cell(row.get(k)) for k in latex_columns)+r' \\'+'\n' for row in rows)
    content+=r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
    (directory/(stem+'.tex')).write_text(content)
    return md


def report(binding):
    root=Path(binding['output_root']);free_space(root)
    science=optional(root/'result_store.json',{'status':'PARTIAL_DEPENDENCY_BLOCK',
        'scene_metrics':[r for rows in binding['baseline_rows'].values() for r in rows.values()],
        'pooled_metrics':binding['baseline_pools'],'scene_method_coverage':52,'full_cohort_pool_coverage':4})
    diagnostics=optional(root/'diagnostics/summary.json',{'status':'MISSING','coverage':{},'comparisons':[],'historical_to_matched':[]})
    selection=optional(root/'selection.json',{'status':'MISSING','selected':'SU01_G1','passing_candidates':[],
        'flags':{},'target_met':False,'material_target_met':False,'deployment':'N0_UNCHANGED'})
    costs=optional(root/'costs/summary.json',{'status':'MISSING','methods':{},'acquisition':{}})
    required={'science':science['status']=='SCIENCE_COMPLETE','diagnostics':diagnostics['status']=='DIAGNOSTICS_COMPLETE',
        'selection':selection['status'] in ('COMPLETE_TARGET_MET','COMPLETE_NO_TARGET_GAIN'),'costs':costs['status']=='COSTS_COMPLETE'}
    complete=all(required.values());methods=[m['id'] for m in binding['specification']['methods']]
    primary=[];supplement=[];coverage=[];contrasts=[]
    for m in methods:
        p={'Method':m,'Rule':METHOD_NAMES[m]};sup={'Method':m}
        for cohort,label in [('replica8','Replica'),('scannet_cf18','CF18')]:
            pool=science['pooled_metrics'].get(cohort,{}).get(m);metrics=pool['metrics'] if pool else {}
            for key in ('apall','ap50','miou'):p[label+'_'+key+'_pct']=None if key not in metrics else 100*metrics[key]
            for key in ('apall','ap50','ap25','miou','macc'):sup[label+'_'+key+'_fraction']=metrics.get(key)
            c=diagnostics['coverage'].get(cohort,{}).get(m)
            baseline=m in ('SU00_D2','SU01_G1')
            coverage.append({'Cohort':cohort,'Method':m,'Selected':None if c is None else c['selected'],
                'Eligible':None if c is None else c['common'],'Applied_changes':0 if baseline else (None if c is None else c['applied_changes']),
                'Proposed_changes':0 if baseline else (None if c is None else c['proposed_changes']),
                'Protected':None if c is None else c['protected'],'Unavailable':None if c is None else c['unavailable'],
                'FULL_A_views':None if c is None else c['successful_FULL_A_views'],
                'Paired_FULL':None if c is None else c['common_FULL_pairs'],
                'N_available':None if c is None else c['source_available_counts'].get('N',0),
                'Q_available':None if c is None else c['source_available_counts'].get('Q',0),
                'F_available':None if c is None else c['source_available_counts'].get('F',0),
                'CPU_ms_scene':costs['methods'].get(m,{}).get('mean_ms_per_scene') if cohort=='replica8' else None,
                'CPU_null_reason':'only Replica8 timed' if cohort!='replica8' else None,
                'Evidence':('Inherited output; literal pass-through' if baseline else
                    'N/Q/F + all same-view A/C; historical class replay' if m in ('SU02_HARD_MATCHED','SU03_STABLE_MATCHED') else
                    'N/Q/F + same-view C; A establishes shared availability' if m=='SU06_F_COARSE' else
                    'N/Q/F + paired A-C' if m=='SU08_PAIRED_DELTA' else
                    'N/Q/F + A; C establishes shared availability'),
                'Availability_null_reason':'baseline does not use update eligibility' if baseline else None})
        p['Target_met']=selection['flags'].get(m,{}).get('target_met')
        p['Material_met']=selection['flags'].get(m,{}).get('material_target_met')
        sup.update(selection['flags'].get(m,{}));primary.append(p);supplement.append(sup)
    for a,b in binding['specification']['comparisons']:
        for cohort in binding['cohorts']:
            d=next((d for d in diagnostics['comparisons'] if (d['cohort'],d['candidate'],d['reference'])==(cohort,a,b)),None)
            delta={} if d is None or d['metric_deltas_fraction'] is None else d['metric_deltas_fraction']
            matches={} if d is None else d['released_matches']['0.5'];sem={} if d is None else d['semantic_outcomes']['0.5']
            contrasts.append({'Cohort':cohort,'Contrast':a+' - '+b,
                **{'Delta_'+k+'_pp':None if k not in delta else 100*delta[k] for k in ('apall','ap50','miou')},
                'GT50_gained':None if d is None else len(matches['new_unique_GT']),
                'GT50_lost':None if d is None else len(matches['lost_unique_GT']),
                'Wrong_to_right':None if d is None else sem.get('WRONG_TO_RIGHT',0),
                'Right_to_wrong':None if d is None else sem.get('RIGHT_TO_WRONG',0),
                'Output_identity_tie':None if d is None else d['output_identity_tie_all_scenes']})
    science_identity=science.get('science_identity',science.get('identity'))
    canonical=seal({**science,'science_identity':science_identity,'selection':selection,'diagnostics':diagnostics,
        'costs':costs,'main_tables':{'table1':primary,'table2':contrasts,'table3':coverage},
        'supplementary_metrics_and_exact_gates':supplement,'report_status':'TABLES_COMPLETE' if complete else 'PARTIAL_DEPENDENCY_BLOCK',
        'required_outputs':required,'publication_status':'EXTERNAL_RECEIPT_REQUIRED'})
    atomic_write_json(root/'result_store.json',canonical)
    table_root=root/'tables'
    t1cols=['Method','Rule','Replica_apall_pct','Replica_ap50_pct','Replica_miou_pct','CF18_apall_pct','CF18_ap50_pct','CF18_miou_pct','Target_met','Material_met']
    t2cols=list(contrasts[0]);t3cols=['Cohort','Method','Selected','Eligible','Applied_changes','Paired_FULL','CPU_ms_scene','Evidence']
    table_text=[table_files(table_root,'table1_primary',canonical['main_tables']['table1'],t1cols,
        'Nine fixed conditions; matched controls explicitly marked. Released ordered pools, percent; gates use unrounded fractions.'),
        table_files(table_root,'table2_contrasts',canonical['main_tables']['table2'],t2cols,
        'Six fixed contrasts in both complete cohorts. Deltas in percentage points; fixed original P semantic outcomes at strict IoU > .50.'),
        table_files(table_root,'table3_coverage_costs',canonical['main_tables']['table3'],t3cols,
        'Common-domain and actual class-change coverage. CPU cost is conditional resident decision time on Replica8 only; shared acquisition is separate.')]
    # Machine table3 retains all availability/exclusion columns, alongside compact display.
    atomic_write_json(table_root/'table3_coverage_costs.json',{'caption':'Full coverage and conditional costs','rows':coverage})
    full=io.StringIO();w=csv.DictWriter(full,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    (table_root/'table3_coverage_costs.csv').write_text(full.getvalue())
    atomic_write_json(table_root/'supplementary_metrics.json',supplement)
    columns=sorted(set(k for row in supplement for k in row));full=io.StringIO();w=csv.DictWriter(full,fieldnames=columns);w.writeheader();w.writerows(supplement)
    (table_root/'supplementary_metrics.csv').write_text(full.getvalue())
    artifact=REPO/ARTIFACT;artifact.mkdir(parents=True,exist_ok=True)
    for p in sorted(table_root.iterdir()):shutil.copyfile(p,artifact/p.name)
    gz_json(artifact/'result_store.json.gz',canonical)
    gz_json(artifact/'source_binding.json.gz',binding)
    # This is a superseded generated file in the new artifact directory only.
    (artifact/'source_binding.json').unlink(missing_ok=True)
    index=ConsumptionIndex(root/'publication/verifications.json');dependencies={};score_arrays={};decision_arrays={};evidence={};eligibilities={};locks={};external_predictions={}
    def dependency(item):
        if 'path' in item and 'sha256' in item:dependencies[item['path']]=item
    for scene in [s for names in binding['cohorts'].values() for s in names]:
        mp=root/'evidence'/scene/'manifest.json';dp=root/'eligibility'/(scene+'.json');lp=root/'predictions'/scene/'receipt.json'
        if mp.exists():
            manifest=read(mp);evidence[scene]=manifest
            for item in manifest['dependencies']:dependency(item)
            for o,row in manifest['incumbents'].items():
                for source,values in row['scores'].items():
                    if values is not None:score_arrays[scene+'/'+o+'/'+source]=np.asarray(values,np.float64)
                if row['p0'] is not None:score_arrays[scene+'/'+o+'/D2_p0']=np.asarray(row['p0'],np.float64)
                for v in row['full_views']:score_arrays[scene+'/'+o+'/A/'+v['region_id']]=np.asarray(v['scores'],np.float64)
        if dp.exists():eligibilities[scene]=read(dp)
        if lp.exists():
            lock=read(lp);locks[scene]=lock
            for method,item in lock['methods'].items():
                p=Path(item['manifest']);mi=index.identity(p);dependency(mi)
                saved=read(p);ai=index.identity(p.parent/saved['arrays']['path'],saved['arrays']);dependency(ai)
                external_predictions[scene+'/'+method]={'manifest':mi,'arrays':ai,'prediction_key':item['prediction_key'],
                    'external_relative_identifier':str(p.relative_to(root)) if p.is_relative_to(root) else 'parent/'+scene+'/'+method}
        scene_dir=artifact/'scenes'/scene;scene_dir.mkdir(parents=True,exist_ok=True)
        ds={m:read(root/'decisions'/scene/(m+'.json')) for m in methods if (root/'decisions'/scene/(m+'.json')).exists()}
        packed=pack_vectors(ds,decision_arrays)
        restored=unpack_vectors(packed,decision_arrays)
        if restored!=ds:raise ValueError('lossless decision roundtrip failed: '+scene)
        for value in restored.values():_verified_identity(value)
        gz_json(scene_dir/'decisions.json.gz',seal({'status':'COMPACT_LOSSLESS_VECTOR_REFERENCES',
            'array_file':'../../decision_vectors.npz','original_decision_identities':{m:d['identity'] for m,d in ds.items()},
            'decisions':packed}))
        diag=root/'diagnostics'/scene/'receipt.json'
        if diag.exists():gz_json(scene_dir/'diagnostics.json.gz',read(diag))
        cs={}
        for p in sorted((root/'coarse'/scene).glob('*/receipt.json')):
            receipt=read(p);cs[p.parent.name]=receipt
            for item in receipt['dependencies']:dependency(item)
            for rid,item in receipt['records'].items():
                if item['available']:score_arrays[scene+'/C/'+rid]=np.asarray(item['scores'],np.float64)
        gz_json(scene_dir/'coarse.json.gz',cs)
    gz_json(artifact/'paired_evidence_manifest.json.gz',evidence)
    gz_json(artifact/'eligibility.json.gz',eligibilities);gz_json(artifact/'prediction_locks.json.gz',locks)
    np.savez_compressed(artifact/'source_scores.npz',**score_arrays)
    np.savez_compressed(artifact/'decision_vectors.npz',**decision_arrays)
    for p in [root/'input_verifications.json',root/'coarse/verifications.json',*sorted((root/'inputs').glob('*/verifications.json'))]:
        if p.exists():
            for entry in read(p)['entries']:dependency(entry['identity'])
    dependency(binding['parent_source_binding']);dependency(binding['parent_result_store']);dependency(binding['parent_publication'])
    atomic_write_json(artifact/'dependency_manifest.json',seal({'dependencies':sorted(dependencies.values(),key=lambda x:x['path']),
        'prediction_references':external_predictions,
        'licensed_RGBD_and_weights':'external dependencies only; no data copies published',
        'large_predictions':'external exact SHA256 plus actual owner/semantic/rank identities in prediction_locks',
        'regeneration_command':full_command(binding),'result_store_identity':canonical['identity']}))
    gz_json(artifact/'execution_costs.json.gz',{'costs':costs,
        'execution':{p.name:read(p) for p in sorted((root/'execution').glob('*.json'))}})
    for name in ('next_stage_assessment.json','freeze.json','tests_initial.log','tests_selection_after_fix.log','initial_state.json'):
        p=root/name
        if p.exists():shutil.copyfile(p,artifact/name)
    for name in ('packaging_correction.json','original_pre_outcome_freeze.json','tests_compact_green.log'):
        p=root/name
        if p.exists():shutil.copyfile(p,artifact/name)
    docs=REPO/REPORT_DOCS;docs.mkdir(parents=True,exist_ok=True)
    acquisition=costs.get('acquisition',{});counts=acquisition.get('successful_counts',{});failed_counts=acquisition.get('failed_counts',{})
    scope=f"Execution: {canonical['status']}; reporting: {canonical['report_status']}. Scene-method rows {canonical['scene_method_coverage']}/234; complete pools {canonical['full_cohort_pool_coverage']}/18.\n"
    limits='Replica8 and CF18 were previously exposed. CF18 comprises 18 captures from seven physical families; this is neither untouched confirmation nor full ScanNet200 validation. No significance or monotone-accuracy claim is supported. Deployment: N0_UNCHANGED.\n'
    cost_text=(f"New successful FC image encodings: {counts.get('new_FC_image_inputs',0)}; successful coarse FULL pools: {counts.get('successful_coarse_pools',0)}; "
        f"unavailable coarse pools: {counts.get('unavailable_coarse_pools',0)}; failed attempts: {acquisition.get('failed_attempt_count',0)}. "
        f"Failed-attempt FC encodings: {failed_counts.get('new_FC_image_inputs',0)}; failed-attempt encoder calls: {failed_counts.get('FC_encoding_attempts',0)}; "
        f"failed-attempt coarse pools: {failed_counts.get('coarse_pool_attempts',0)}. "
        f"AnyUp QK, new geometric projections, N/Q/frontend inference and end-to-end cold calls: 0. "
        f"Coarse worker wall (including resume invocations): {acquisition.get('worker_wall_seconds_including_resume_invocations')} s; "
        f"model-load wall: {acquisition.get('model_load_seconds')} s. Model-load time is included in worker/first-leaf wall, not additive. "
        'A and historical FC sources are reused, not free at deployment. CPU timing excludes loading, GPU models, payload/ranks, evaluation and serialization.\n')
    if costs['status']=='MISSING':cost_text='Acquisition work and CPU timing are unmeasured; missing values are not zero cost.\n'
    status=f"Selection: {selection['status']}; selected {selection['selected']}; passing candidates {selection['passing_candidates']}; material target {selection.get('material_target_met',False)}.\n"
    (docs/'SOURCE_UPDATE_RESULTS.md').write_text('# Source-preserving semantic update results\n\n'+scope+'\n'+status+'\n'+ '\n\n'.join(table_text)+'\n\n'+cost_text+'\n'+limits+
        '\nAP25/mAcc and exact gates: supplementary_metrics.csv/json. Full per-scene, GT50/75, tied TP/FP multiplicity, rank changes and class deltas are in the compact result/diagnostic bundles. Probability equality is separate from label/output equality.\n')
    parent=read(binding['parent_result_store']['path'])
    historical='\nHistorical unrestricted IR06/IR07 context (not matched main-table results; percent):\n\n| Cohort | Historical method | APall | AP50 | AP25 | mIoU | mAcc | Source pool identity |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for cohort in binding['cohorts']:
        for m in ('IR06_ANYUP_REREAD','IR07_BOUNDARY_STABLE'):
            pool=parent['pooled_metrics'][cohort][m]
            historical+='| '+cohort+' | '+m+' | '+' | '.join(f"{100*pool['metrics'][k]:.3f}" for k in ('apall','ap50','ap25','miou','macc'))+' | '+pool['identity']+' |\n'
    with (docs/'SOURCE_UPDATE_RESULTS.md').open('a') as handle:handle.write(historical)
    (docs/'SOURCE_UPDATE_SELECTION.md').write_text('# Exact exposed-cohort selection\n\n'+status+
        '\nAll five metrics in both cohorts must be at least G1 minus 1e-10; CF18 APall must exceed D2 by more than 1e-10, with AP50 at least D2 minus 1e-10. The material marker adds 0.001 fraction. Lexicographic tie metrics and simplicity order are the verbatim protocol.\n\n'+
        '```json\n'+json.dumps(selection,indent=2,ensure_ascii=False)+'\n```\n\n'+limits)
    (docs/'SOURCE_UPDATE_CLAIMS.md').write_text('# Evidence boundaries\n\n'+scope+'\n'+status+'\n'+limits+
        '\nThe fixed G1 owner arrays, recovered classes and raw-zero protected incumbent classes remain unchanged. Only common-domain original incumbents can change class; official current-class ranks are recomputed, including recovered owners. Hard/stable controls are matched replays, not copied unrestricted IR06/IR07 headline results.\n\n'+cost_text+
        '\nNo standalone latency, GPU-memory advantage, online 30 FPS, universal nondegradation, independent evidence from N/Q, novel Bayesian rule, learned risk gate or structural recovery improvement is claimed. Null cells denote absent evidence or an untimed cohort, never zero cost.\n')
    (docs/'SOURCE_UPDATE_HANDOFF.md').write_text('# Reproducible handoff\n\n'+scope+'\n'+status+
        f"\nParent store {binding['parent_store_identity']}; source binding {binding['identity']}; science identity {science_identity}; result-store identity {canonical['identity']}.\n\n"+
        '```bash\n'+full_command(binding)+'\n```\n\n'+
        'Use the inherited controller/FC environments and external dependency_manifest.json. --storage-root, --path-map and --gpu provide explicit relocation. Resume validates content and archives only changed new-task descendants. Original experiments are read-only.\n\n'+
        'Exactly three main table families are generated from result_store.json. source_scores.npz preserves precise real N/Q/F/D2/A/C arrays; paired_evidence_manifest and per-scene decisions/coarse/diagnostics retain masks, identities and full distributions. decision_vectors.npz deduplicates float64 vectors losslessly; reporting.unpack_vectors restores per-scene decisions and verifies original vector and decision identities. source_binding.json.gz is the exact source binding. Large payloads/dense/RGBD/weights remain external.\n\n'+
        'The packaging-only post-outcome correction and original pre-outcome freeze are published separately. Scientific producers, predictions, scores and measured CPU samples were retained exactly; no result-conditioned method or timing change was made.\n\n'+
        'Historical unrestricted IR06/IR07 replay and their original-to-matched metric/coverage differences are in result_store.diagnostics.historical_to_matched, with actual prediction and scoring identities. No historical score is substituted for a new matched method.\n\n'+
        'Publication proof is external publication/final.json; it records the full local/remote commit hashes and observed full-command exit. No commit contains its own commit hash.\n\n'+limits)
    size=sum(p.stat().st_size for p in artifact.rglob('*') if p.is_file())
    if size>=binding['specification']['resources']['publish_target_MiB']*2**20:
        raise ValueError(f'compact artifact size {size/2**20:.2f} MiB exceeds 30 MiB')
    result=seal({'status':'TABLES_COMPLETE' if complete else 'PARTIAL_DEPENDENCY_BLOCK','result_store_identity':canonical['identity'],
        'science_identity':science_identity,'artifact_bytes':size,'artifact_root':str(ARTIFACT),'table_families':3,'reports':binding['specification']['publication']['four_reports'],
        'required_outputs':required})
    atomic_write_json(root/'tables/receipt.json',result);index.write_memo(root/'publication/verifications.json');return result


def full_command(binding):
    return (binding['specification']['controller_python']+' scripts/evaluation/run_ovimap_source_preserving_update.py '
        '--spec configs/static_ovmap/source_preserving_update_v1.json --parent-root '+binding['parent_logical_root']+
        ' --output-root '+binding['logical_root']+' --phase all --resume')


def publish(binding):
    root=Path(binding['output_root']);table=read(root/'tables/receipt.json')
    allowed=['configs/static_ovmap/source_preserving_update_v1.json',str(DOCS),str(ARTIFACT),
        'docs/superpowers/plans/2026-10-08-source-preserving-update.md','scripts/evaluation/run_ovimap_source_preserving_update.py',
        'src/static_ovmap/source_preserving_update','tests/evaluation/test_source_preserving_update.py',
        'tests/evaluation/test_source_preserving_update_compact.py']
    allowed.extend(str(REPORT_DOCS/name) for name in binding['specification']['publication']['four_reports'])
    def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
    if git('branch','--show-current')!=binding['branch']:raise ValueError('publication branch changed')
    subprocess.run(['git','diff','--check','--',*allowed],cwd=REPO,check=True)
    subprocess.run(['git','add','--',*allowed],cwd=REPO,check=True)
    # Never incorporate unrelated staged work into this authorized publication.
    staged=git('diff','--cached','--name-only').splitlines()
    if any(not any(p==a or p.startswith(a+'/') for a in allowed) for p in staged):raise ValueError('unrelated staged changes require separate integration')
    if staged:
        subprocess.run(['git','commit','-m','Evaluate fixed source-preserving semantic updates on Replica8 and CF18'],cwd=REPO,check=True)
    local=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+binding['branch']],cwd=REPO,check=True)
    remote=git('ls-remote','origin','refs/heads/'+binding['branch']).split()[0]
    if local!=remote:raise ValueError('ordinary push did not produce full matching local/remote SHA')
    receipt=seal({'status':'PUSH_VERIFIED','local_HEAD':local,'remote_HEAD':remote,
        'UTC':datetime.now(timezone.utc).isoformat(),'branch':binding['branch'],
        'table_status':table['status'],'result_store_identity':table['result_store_identity'],
        'science_identity':table['science_identity'],'full_resume_exit_code':None,
        'exit_code_reason':'must be filled by the external observer after the full CLI reaches a terminal exit',
        'deployment':'N0_UNCHANGED'})
    atomic_write_json(root/'publication/final.json',receipt);return receipt
