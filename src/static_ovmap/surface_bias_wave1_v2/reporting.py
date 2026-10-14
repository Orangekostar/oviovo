"""Tables come from sealed observations; missing science stays null."""
import csv
import gzip
import io
import json
from pathlib import Path
import shutil
from .common import REPO, BASE, METHODS, LAMBDAS, read,verified,write,digest_file,source_digest,atomic_text

ART='artifacts/static_ovmap/surface_bias_wave1_v2'
DOC='docs/paper/static_ovmap/surface_bias_wave1_v2'

def optional(root,name):
    path=Path(root)/name
    return verified(path) if path.exists() else None

def table(folder,name,rows):
    folder=Path(folder); columns=list(dict.fromkeys(k for r in rows for k in r))
    write(folder/(name+'.json'),dict(rows=rows,columns=columns,units='metrics FRACTION; display percentages explicit in *_pct'))
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n'); writer.writeheader(); writer.writerows(rows); atomic_text(folder/(name+'.csv'),stream.getvalue())
    cell=lambda value: '—' if value is None else str(value).replace('|','\\|').replace('\n',' ')
    content='| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+''.join('| '+' | '.join(cell(r.get(c)) for c in columns)+' |\n' for r in rows)
    atomic_text(folder/(name+'.md'),content)

def science_status(diagnostic,activation,evaluation):
    if activation is None or activation.get('status')!='COMPLETE': return 'UNASSESSED_DEPENDENCY'
    if evaluation and evaluation.get('status')=='EVALUATION_COMPLETE':
        gains=evaluation['map_gains']; m1=METHODS[2]
        if gains.get(m1,{}).get('MAP_GAIN_VS_FG2'): return 'REGRESSION_MAP_GAIN'
        if any(row['MAP_GAIN_VS_FG2'] for row in gains.values()): return 'ROUTINE_CONTROL_GAIN_ONLY'
        return 'MIXED_MAP_RESULT'
    if activation['expanded']: return 'READOUT_GAIN_MAP_UNCONFIRMED'
    return 'SIGNAL_ONLY_NO_READOUT_GAIN' if diagnostic and diagnostic.get('mechanism_signal') else 'NO_HELDOUT_SIGNAL'

def execution_complete(*,parity,diagnostic,selection,repeat,prefix,activation,evaluation):
    if not all(value and value.get('status')=='COMPLETE' for value in (parity,diagnostic,selection,repeat,prefix,activation)):
        return False
    return bool(evaluation and evaluation.get('status')==('EVALUATION_COMPLETE' if activation['expanded'] else 'NOT_ACTIVATED_BY_FIXED_RULE'))

def compact_results(root,art,evaluation):
    """Publish readouts and sufficient statistics, never grids, GT or point maps."""
    root=Path(root); art=Path(art); folder=art/'compact'; folder.mkdir(parents=True,exist_ok=True)
    files=[]; external=[]
    def compressed(name,value):
        dest=folder/name; dest.parent.mkdir(parents=True,exist_ok=True)
        raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        temporary=dest.with_name(dest.name+'.tmp'); temporary.write_bytes(gzip.compress(raw,mtime=0)); temporary.replace(dest)
        files.append(str(dest.relative_to(art)))
    for source in sorted((root/'recognition').glob('*.json')):
        value=verified(source)
        if value.get('status')!='COMPLETE' or 'arrays' not in value: continue
        array=value['arrays']; actual=Path(array['path'])
        if not actual.exists(): raise ValueError('Compact recognition output is missing: '+str(actual))
        if digest_file(actual)!=array['sha256']: raise ValueError('Compact recognition bytes changed')
        destination=folder/'readouts'/actual.name; destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(actual,destination)
        value=dict(value,runtime_receipt_identity=value['identity'],arrays=dict(array,path=str(destination.relative_to(art))))
        value.pop('identity',None); write(destination.with_suffix('.json'),value)
        files.extend([str(destination.relative_to(art)),str(destination.with_suffix('.json').relative_to(art))])
        external.append(dict(array,regenerate_phase='parity/select/repeat/prefix',kind='READOUT_ARRAY'))
    readout_units=[]
    for schema in sorted((root/'recognition/units').iterdir()) if (root/'recognition/units').exists() else []:
        if not schema.is_dir(): continue
        for path in sorted(schema.glob('*.json')):
            unit=verified(path)
            readout_units.append(dict(schema=schema.name,key=path.stem,record=unit))
            if unit.get('arrays'): external.append(dict(unit['arrays'],regenerate_phase='parity/select/repeat/prefix',kind='CONDITION_READOUT_ARRAY',input_identity=unit.get('input_identity')))
    diagnostic_units=[]
    for role in ('train','dev'):
        diagnostic_units.extend(verified(p) for p in sorted((root/'diagnostics'/role).glob('*.json')))
    if diagnostic_units: compressed('diagnostic_units.json.gz',diagnostic_units)
    if readout_units: compressed('readout_units.json.gz',readout_units)
    # Preserve partial prediction traces too, while keeping partial maps out of
    # formal cohort pools. Each trace is output scores/labels, not raw inputs.
    for scene in sorted((root/'decisions').iterdir()) if (root/'decisions').exists() else []:
        if not scene.is_dir(): continue
        for path in sorted(scene.glob('*.json')): compressed('decisions/'+scene.name+'/'+path.name+'.gz',verified(path))
    if evaluation:
        write(folder/'map_summary.json',evaluation); files.append('compact/map_summary.json')
        unique={}
        for row in evaluation.get('scene_metrics',[]):
            source=Path(row['evaluation_receipt']); key=row['evaluation_identity']
            if key in unique: continue
            score=read(source)
            if score['status']!='COMPLETE' or score['identity']!=key: raise ValueError('Compact scoring receipt differs')
            compressed('scores/'+key+'.json.gz',score)
            matches=source.with_name('matches.json.gz')
            if not matches.exists(): raise ValueError('Official AP sufficient statistics missing: '+str(matches))
            dest=folder/'scores'/(key+'_matches.json.gz'); shutil.copyfile(matches,dest); files.append(str(dest.relative_to(art)))
            unique[key]=dict(scene=row['scene'],receipt='compact/scores/'+key+'.json.gz',matches=str(dest.relative_to(art)),source_sha256=digest_file(matches))
        if unique:
            write(folder/'scorer_index.json',dict(status='COMPLETE',scene_inputs=unique,regenerate='trace_released_matches on the archived matches; sum archived confusion in official scene order; evaluator code is fixed by each context')); files.append('compact/scorer_index.json')
        for cohort,methods in evaluation.get('pooled_metrics',{}).items():
            for method,pool in methods.items():
                if pool.get('per_class_receipt'): compressed('per_class/'+cohort+'/'+method+'.json.gz',verified(pool['per_class_receipt']))
    if external: compressed('external_intermediates.json.gz',external)
    write(art/'compact_results_index.json',dict(status='MEASURED_COMPACT_OUTPUTS' if files else 'NO_NEW_SCIENTIFIC_OUTPUTS',published_files=files,external_intermediate_count=len(external),external_intermediates='compact/external_intermediates.json.gz' if external else None,excluded='RAW_RGB_DEPTH_GT_POINT_MAPS_DENSE_GRIDS_AND_FOUNDATION_WEIGHTS',recompute_tables='Use archived sealed summary/table JSON; use diagnostic units to sum N/D and readout records to summarize AMC; runtime report regenerates all table formats'))

def run(root,*,execution_status=None,error=None):
    root=Path(root); art=REPO/ART; docs=REPO/DOC; art.mkdir(parents=True,exist_ok=True); docs.mkdir(parents=True,exist_ok=True)
    parity=optional(root,'parity.json'); diagnostic=optional(root,'diagnostic_summary.json'); selection=optional(root,'selection_seed17.json'); repeat=optional(root,'repeat_seed29.json'); prefix=optional(root,'prefix.json'); activation=optional(root,'activation.json'); evaluation=optional(root,'evaluation/store.json'); deps=optional(root,'dependency_manifest.json'); stages=optional(root,'stage_exits.json'); publication=optional(root,'publication/final.json')
    if execution_status is None: execution_status='COMPLETE' if execution_complete(parity=parity,diagnostic=diagnostic,selection=selection,repeat=repeat,prefix=prefix,activation=activation,evaluation=evaluation) else 'BLOCKED_DEPENDENCY' if deps and deps.get('missing') else 'PARTIAL_ENGINEERING'
    science=science_status(diagnostic,activation,evaluation)
    publication_status='PUSH_VERIFIED' if publication and publication.get('status')=='PUSH_VERIFIED' and publication.get('release_identity',{}).get('scientific_source_digest')==source_digest() else 'PUBLICATION_BLOCKED'
    summary=dict(execution_status=execution_status,science_status=science,publication_status=publication_status,publication_snapshot='NOT_ATTEMPTED_OR_BLOCKED_AT_REPORT_SNAPSHOT' if publication_status!='PUSH_VERIFIED' else 'VERIFIED_EXTERNAL_RECEIPT',task_id='fg2-surface-bias-wave1-v2',BASE=BASE,scientific_source_digest=source_digest(),mechanism_signal=diagnostic.get('mechanism_signal') if diagnostic else None,pass_both=activation.get('pass_both') if activation else None,expanded=activation.get('expanded') if activation else None,map_gains=evaluation.get('map_gains') if evaluation else None,missing=deps.get('missing',[]) if deps else [],error=error,new_optimizer_updates=0,new_rgb_encodings=0,new_sam_proposals=0,deployment='N0_UNCHANGED')
    write(root/'run_summary.json',summary); write(art/'run_summary.json',summary)
    rows=[dict(test='REAL_FC_PARITY',status=parity['status'] if parity else 'NOT_RUN',original_objects=parity.get('dev_originals') if parity else None,reason='Exact parent FG/LR dependencies unavailable' if parity is None else None)]
    table(art/'tables','table1_baseline_checks',rows)
    rows=[]
    if diagnostic:
        for role,conditions in diagnostic['summary'].items():
            for condition,values in conditions.items():
                for method,stats in values['own'].items(): rows.append(dict(role=role,condition=condition,method=method,R=stats['R_family_mean'],objects=stats['objects'],families=stats['families'],N=stats['N'],D=stats['D'],support=stats['support']))
    else: rows=[dict(status='NOT_RUN',R=None,reason='P0 parent dependency unavailable; no held-out mechanism conclusion')]
    table(art/'tables','table2_diagnostics',rows)
    rows=[]
    for method in METHODS:
        for strength in LAMBDAS:
            values=next((r for r in selection['candidates'][method] if r['lambda_']==strength),{}) if selection else {}
            rows.append(dict(method=method,seed=17,lambda_=strength,status='MEASURED' if values else 'NOT_RUN',A_pct=100*values['A'] if values else None,M_pct=100*values['M'] if values else None,C_pct=100*values['C'] if values else None,net_corrections=values.get('net_corrections'),numeric_fallback_rate=values.get('numeric_fallback_rate'),selected=selection['selected'][method]['lambda_']==strength if selection else None))
        values=repeat['rows'][method] if repeat else {}
        rows.append(dict(method=method,seed=29,lambda_=selection['selected'][method]['lambda_'] if selection else None,status='MEASURED_FIXED_REPEAT' if values else 'NOT_RUN_NOT_LOCKED',A_pct=100*values['A'] if values else None,M_pct=100*values['M'] if values else None,C_pct=100*values['C'] if values else None,net_corrections=values.get('net_corrections'),numeric_fallback_rate=values.get('numeric_fallback_rate'),selected=None))
    table(art/'tables','table3_selection_repeat',rows)
    map_rows=[]
    if evaluation and evaluation.get('status')=='EVALUATION_COMPLETE':
        for cohort,methods in evaluation['pooled_metrics'].items():
            for method,row in methods.items(): map_rows.append(dict(cohort=cohort,method=method,status='OFFICIAL_ORDERED_POOL',**{k+'_pct':None if v is None else 100*v for k,v in row['metrics'].items()}))
    else: map_rows=[dict(status=evaluation['status'] if evaluation else 'NOT_RUN_DEPENDENCY',APall_pct=None,AP50_pct=None,AP25_pct=None,mIoU_pct=None,mAcc_pct=None,reason='P3 not assessed' if activation is None else 'Fixed DEV-repeat rule inactive' if not activation['expanded'] else 'Activated map not complete')]
    table(art/'tables','table4_whole_map',map_rows)
    resources=[]
    for p in sorted((root/'resources').glob('*.json')) if (root/'resources').exists() else []:
        value=verified(p); resources.append(value)
    table(art/'tables','table5_cost_scope',resources or [dict(boundary='NO_REAL_FC_CACHE_EXPERIMENT_EXECUTED',new_optimizer_updates=0,new_rgb_encodings=0,new_sam_proposals=0,new_phi_calls=0,new_map_scorer_calls=0,GPU_peak_bytes=None,wall_seconds=None,online_FPS=None)])
    table(art/'tables','prefix',prefix['rows'] if prefix else [dict(status='NOT_RUN',prefix=None,reason='Selection not available')])
    for name in ('dependency_manifest.json','parity.json','diagnostic_summary.json','selection_seed17.json','repeat_seed29.json','prefix.json','activation.json','stage_exits.json','storage_preflight.json','tests.json'):
        source=root/name
        if source.exists(): shutil.copyfile(source,art/name)
        elif name not in ('tests.json','storage_preflight.json'): write(art/name,dict(status='NOT_RUN_DEPENDENCY',reason='Exact numerical prerequisites unavailable',numeric_values=None))
    spec=read(REPO/'configs/static_ovmap/surface_bias_wave1_v2.json')
    shutil.copyfile(REPO/'configs/static_ovmap/surface_bias_wave1_v2.json',art/'surface_bias_wave1_v2.json')
    registry=dict(status='LOCKED_CALIBRATION' if selection else 'NO_SELECTED_MODEL',BASE=BASE,parent_heads=spec['heads'],no_new_weights=True,lambdas={m:selection['selected'][m]['lambda_'] for m in METHODS} if selection else None,solver_schema=2,scientific_source_digest=source_digest())
    write(art/'selected_model_registry.json',registry)
    write(art/'resources.json',dict(actual_phase_resources=resources,new_optimizer_updates=0,new_rgb_encodings=0,new_sam_proposals=0,online_FPS=None,nested_timers_not_additive=True))
    compact_results(root,art,evaluation)
    # Published parent values are context, never observations from this task.
    parent_table=REPO/'artifacts/static_ovmap/frozen_fc_routing_v1/tables/table1_whole_map.json'
    inherited=read(parent_table); write(art/'inherited_FG_results.json',dict(status='INHERITED_PUBLISHED_CONTEXT_ONLY',BASE=BASE,source_path=str(parent_table.relative_to(REPO)),source_sha256=digest_file(parent_table),table=inherited,new_measurement=False))
    reason='; '.join(summary['missing']) or error or 'See formal phase records'
    bound=optional(root,'binding.json') or {}; actual_parent=bound.get('fg_parent_root',(deps or {}).get('checked_parent','/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001'))
    gpu=bound.get('gpu','0')
    table_links='\n'.join(f'- [Table {i}]({(Path("../../../")/ART/"tables"/name).as_posix()}.md)' for i,name in enumerate(('table1_baseline_checks','table2_diagnostics','table3_selection_repeat','table4_whole_map','table5_cost_scope'),1))
    # Links from this nested docs directory need four levels to repository root.
    table_links=table_links.replace('../../../artifacts/','../../../../artifacts/')
    atomic_text(docs/'RESULTS.md',f'''# FG2 surface bias wave1 v2 results

Execution: **{execution_status}**. Science: **{science}**. Deployment: N0_UNCHANGED.

{reason}

New optimizer updates / RGB encodings / SAM proposals: **0 / 0 / 0**. Missing experiment metrics remain null. Existing FG2 results are separately identified as inherited published context; they do not establish a gain for this new calibration. Cached-grid replay does not establish online 30 FPS or an efficiency advantage over OVI-MAP.

{table_links}

All requested lambda candidates and the fixed-repeat status are retained. Seed29 and map labels never select lambda. CF18 contains18 captures from7 physical families and is not the complete ScanNet200 validation set. The parent proposals were derived from GT instances, so label-free bias fitting does not make the entire pipeline GT-free or training-free.

The gauge identifies relative offsets only. Equal view pooling with complete visibility can cancel the correction exactly; phi cannot manufacture a change. Independent-support nulls can degenerate. Spatial folds with bilinear-donor purge are not independent datasets because receptive fields may overlap.
''')
    atomic_text(docs/'CLAIMS.md','''# Evidence boundaries

Implemented hypothesis: estimate per-view relative raw-feature offsets from matched physical sites and intervene on values while keeping the frozen FG2 key/quality/query branch unchanged.

Allowed claims require measured evidence. A positive held-out residual statistic alone is neither statistical significance nor a causal demonstration. Map gains require both cohorts and both locked seeds; superiority over C_STAR requires its full map comparison. Fixed-owner semantic improvements do not establish better instance boundaries.

Improved FG2 calibration requires measured gains against the same-seed parent. This benchmark protocol does not establish overall superiority to OVI-MAP, first-ever multi-view consistency, zero-shot training, independent full ScanNet200 validation, online30FPS, or new trained weights. C1/C2 gains without M1 superiority support routine calibration only.
''')
    atomic_text(docs/'HANDOFF.md',f'''# FG2 calibration handoff

BASE: `{BASE}`. Branch: `research/fg2-surface-bias-wave1-v2` (actual publication branch is in PUBLICATION_RECEIPT).
Execution: {execution_status}; science: {science}; deployment: N0_UNCHANGED.
Scientific source digest: `{source_digest()}`.

Original seed17 step1000 and seed29 step2000 heads are referenced by the [selected model registry](../../../../{ART}/selected_model_registry.json); no head was trained here. Lambda lock: {json.dumps(registry['lambdas'])}. The full parent source binding is verified before any in-memory path migration; parent_root keeps its inherited meaning. FG_ROOT is the CLI parent-root, LR_ROOT comes from that binding. Dependencies: see [dependency manifest](../../../../{ART}/dependency_manifest.json).

Use the existing controller environment and FC_python from the verified binding. Inherit one GPU string; the FC worker uses execution_config(binding)['gpu_lock'] and queued_gpu_lock. All writes belong to OUT; parent caches and upstream sources are read-only. Run from the worktree:

```bash
PY_CTRL=/home/ww/miniconda3/envs/ovimap-map/bin/python
FG_ROOT={actual_parent}
OUT={root}
GPU={gpu}
"$PY_CTRL" -u scripts/evaluation/run_fg2_surface_bias_wave1_v2.py --spec configs/static_ovmap/surface_bias_wave1_v2.json --parent-root "$FG_ROOT" --output-root "$OUT" --gpu "$GPU" --phase all --resume
PYTHONPATH=src "$PY_CTRL" -m pytest -q tests/test_surface_bias_wave1_v2.py
"$PY_CTRL" scripts/evaluation/run_fg2_surface_bias_wave1_v2.py --spec configs/static_ovmap/surface_bias_wave1_v2.json --parent-root "$FG_ROOT" --output-root "$OUT" --phase report --resume
```

Table Markdown/CSV/JSON come from sealed runtime JSON; the report phase regenerates them. Actual scientific worker exits are in stage_exits. The final all-process return code is observed by its invoking parent and stored externally, never predeclared in its own result commit.

Recovery point: restore the exact missing parent manifests, referenced grids, FC/text and input arrays; rerun identical spec/paths. Different scientific parameters require a new version. Do not recompute RGB, train, select another checkpoint, use similar caches or shrink cohorts. Blocked dependency: {reason}.

Results: [RESULTS](RESULTS.md). Limits and supported claims: [CLAIMS](CLAIMS.md). Requirement audit: [REVIEW](REVIEW.md). Large runtime arrays stay in OUT; Git contains compact tables and summaries. Cached replay performance is not online FPS.
''')
    measured=lambda value: 'MEASURED' if value and value.get('status')=='COMPLETE' else 'REAL_EXPERIMENT_BLOCKED'
    checks=[
        ('T0 / sections0–2','Fixed BASE, isolated namespace, original package/config bytes','common.py / binding.py / frozen docs','CODE_REVIEWED'),
        ('T0 / section2.2','P0–P2 exact parents; P3 prerequisites lazy; no AnyUp substitute','binding.minimal_fc_dependencies / test05','SYNTHETIC_DEPENDENCY_SCOPE_VERIFIED'),
        ('T0 / section2.3','Verify parent before path rewrite; preserve inherited parent_root','binding.preflight / consumed_manifest_identities','CODE_REVIEWED_REAL_BINDING_BLOCKED'),
        ('T0 / section2.3','Two fixed head SHAs/steps; FC/text bytes and class order','binding.runtime_registry','REAL_EXPERIMENT_BLOCKED'),
        ('T0 / sections2–3','Read-only links and exact class_split copy; explicit CLI phases','binding.py / runner.py','CODE_REVIEWED'),
        ('T1 / section4','Float64 fixed-scale Huber and zero-mean weighted graph KKT','solver.py / original focused tests01–02','SYNTHETIC_SOLVER_VERIFIED'),
        ('T1 / section4','C1 uses all valid views; C2 per-site Huber; fixed lambda grid','solver.correction / selection.py','CODE_REVIEWED'),
        ('T1 / section4','Only local_raw changes; local_unit/query/quality/eta unchanged','adapter.py / parent-head focused test03','SYNTHETIC_PARENT_HEAD_VERIFIED'),
        ('T1 / section4','Original nonfinite error; only new numeric correction falls back','adapter._adapt','CODE_REVIEWED'),
        ('T1 / sections4–5','Same-seed lambda0 and geometric fallback; no FC8 fallback','adapter.py / prediction.parity','REAL_FC_PARITY_BLOCKED'),
        ('T1 / section4','Relative gauge, balanced-pool cancellation, shuffle degeneracy','solver.py / test02 / CLAIMS','SYNTHETIC_BOUNDARY_VERIFIED'),
        ('T1 / section5','2 TRAIN ×2 seeds ×4 conditions ×2/4/8 real FC checks','prediction.parity / parity.json',measured(parity)),
        ('T1 / section5','69 base DEV keyed historical labels/AMC and embedding tolerance','prediction.compare_history',measured(parity)),
        ('T2 / section6.1','622/69 base objects,24/4 families, four condition-specific positions','binding.py / diagnostics.spatial_folds / test05',measured(diagnostic)),
        ('T2 / section6.1','Four spatial folds and actual bilinear-donor global purge','diagnostics.py / test05','SYNTHETIC_ISOLATION_VERIFIED_REAL_DIAGNOSIS_BLOCKED' if not diagnostic else measured(diagnostic)),
        ('T2 / section6.1','Missing diagnostic metadata unsupported; no fabricated positions','diagnostics.diagnostic_metadata / data.ObjectLoader / test05','SYNTHETIC_PARENT_COMPATIBILITY_VERIFIED'),
        ('T2 / section6.2','Stable five null seeds; degenerate null separately labeled','diagnostics.independent_bias','CODE_REVIEWED_REAL_DIAGNOSIS_BLOCKED' if not diagnostic else measured(diagnostic)),
        ('T2 / section6.2','Sum N/D before R; equal families; paired common support','diagnostics.aggregate / diagnostic_units compact archive',measured(diagnostic)),
        ('T2 / section6.2','Signal coverage floors; signal never gates P2/P3','diagnostics.run / selection.activate',measured(diagnostic)),
        ('T3 / section6.3','Full200 top1/NLL; inherited base-class CE at0.07; exact AMC','prediction.recognition / inherited summarize',measured(selection)),
        ('T3 / section6.3','All12 seed17 candidates; AMC/lambda tie rule; lock C_STAR','selection.lock_selection / test06',measured(selection)),
        ('T4 / section6.4','Seed29 locked repeats only; no champion replacement','prediction.repeat / selection.activate / test06',measured(repeat)),
        ('T4 / section6.4','Exact pass thresholds and three activation branches','selection.passes/activate / test06',measured(activation)),
        ('T4 / section6.5','Both seeds, four conditions,2/4 prefixes; lambda0 aliases','prediction.prefix / test05/07',measured(prefix)),
        ('T4 / section6.5','Group raw/phi/embedding/attention effect and harm audits','adapter.py / prediction.recognition','CODE_REVIEWED_REAL_READOUT_BLOCKED' if not selection else measured(selection)),
        ('T5 / section7.1','All26 scenes and all expanded arms locked before GT','prediction.predict / evaluation.run','REAL_MAP_NOT_REACHED' if not evaluation else evaluation['status']),
        ('T5 / section7.2','Fixed G1 owner/support; replace F only; retain N/Q/temperature/D2','prediction.predict / unchanged build_payload','CODE_REVIEWED_REAL_MAP_BLOCKED'),
        ('T5 / section7.3','Official ordered pools, null class, full200 base/novel, Replica N/A','evaluation.py / compact per-class data','CODE_REVIEWED_REAL_MAP_BLOCKED'),
        ('T5 / section7.3','Identity-proven aliases; exactly2 additional active-run scorer checks','evaluation.evaluate_scene / dynamic coverage','CODE_REVIEWED_REAL_MAP_BLOCKED'),
        ('T5 / section7.4','Four-cell AP/mIoU map gate and M1 vs locked C_STAR','evaluation.run / science_status','CODE_REVIEWED_REAL_MAP_BLOCKED'),
        ('T6 / section8','Exactly8 focused groups; real FC dependency skip is explicit','tests/test_surface_bias_wave1_v2.py / tests.json','FOCUSED_TESTS_RECORDED_REAL_FC_SKIP'),
        ('T6 / section8','Interrupted readout resumes without repeating completed head','common.cached_unit / actual readout-loop test07','SYNTHETIC_INTERRUPTION_VERIFIED'),
        ('T6 / section8','Complete files verified; superseded identities/results retained','common.preserve_unit / test07','SYNTHETIC_RECOVERY_VERIFIED'),
        ('T6 / sections3,8','One queued GPU; FP32;32GiB host budget;3 CPU scorers; space reserve','runner.py / PrefixLoader / common.py / binding.py','CODE_REVIEWED_REAL_RESOURCE_MEASUREMENT_BLOCKED'),
        ('T7 / section9.1','Five tables in MD/CSV/JSON; every finite candidate retained','reporting.table / actual artifact files','GENERATED_FROM_SEALED_JSON'),
        ('T7 / section9.2','Frozen artifact config; compact readouts, N/D, map/match/per-class data','reporting.compact_results / compact_results_index.json','NO_NEW_SCIENTIFIC_OUTPUTS' if not selection and not diagnostic and not evaluation else 'COMPACT_OUTPUTS_ARCHIVED'),
        ('T7 / section9.2','40/150MiB finite task paths; no RGB/GT/grids/base weights','publication.check_files / artifact_index.json','CHECKED_BEFORE_ACTUAL_C1'),
        ('T7 / section9.3','Normal C1/C2 push, prior-release retention, unchanged resume reuse','publication.py / test08 / committed receipt','ACTUAL_NEW_C1_PENDING_AT_REPORT_SNAPSHOT'),
        ('T7 / section9.4','Three status axes; real worker exits; external observed all exit','runner.py / run_summary.json / stage_exits.json','OBSERVED_SCIENTIFIC_FAILURE_RETAINED' if execution_status!='COMPLETE' else 'FORMAL_PHASE_STATUSES_COMPLETE'),
        ('T7 / section10','No unsupported mechanism/map/instance/onlineFPS/OVI-MAP claim','CLAIMS.md / RESULTS.md','EVIDENCE_BOUNDARIES_REVIEWED'),
    ]
    audit=[dict(prompt=section,requirement=requirement,evidence=evidence,status=status) for section,requirement,evidence,status in checks]
    write(art/'requirement_audit.json',dict(review_owner='PRIMARY_AGENT',scientific_source_digest=source_digest(),execution_status=execution_status,science_status=science,checks=audit,remaining_required_real_experiments=summary['missing']))
    atomic_text(docs/'REVIEW.md','# Primary-agent requirement audit\n\n'+f'Execution: **{execution_status}**. Science: **{science}**. The full approved prompt remains the acceptance contract.\n\n| Prompt | Requirement | Evidence | Actual status |\n| --- | --- | --- | --- |\n'+''.join(f'| {section} | {requirement} | {evidence} | {status} |\n' for section,requirement,evidence,status in checks)+'\nThe targeted repair preserves the original published release and the frozen scientific spec. Diagnostic-only field compatibility calls the original parent kernels; it does not invent UV, grids or positions. Missing forward observations still require the exact original tensors or caches. Synthetic tests verify engineering behavior and cannot establish FC parity or new scientific gains.\n\nPhase completion follows the formal receipts above. Actual publication is proven by the committed C1 receipt and external C2 final receipt; this C1 report never predeclares its future push.\n')
    inventory=[]
    for p in sorted(art.rglob('*')):
        if p.is_file() and p.name not in ('artifact_index.json','PUBLICATION_RECEIPT.json'): inventory.append(dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest_file(p)))
    write(art/'artifact_index.json',dict(scope='C1_TASK_ARTIFACTS_EXCLUDING_INDEX_AND_FUTURE_PUBLICATION_RECEIPT',scientific_source_digest=source_digest(),files=inventory,large_runtime_arrays='OUT_ONLY',regenerate='entrypoint --phase report with original runtime arguments'))
    return summary
