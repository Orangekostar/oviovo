"""Tables come from sealed observations; missing science stays null."""
import csv
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

def run(root,*,execution_status=None,error=None):
    root=Path(root); art=REPO/ART; docs=REPO/DOC; art.mkdir(parents=True,exist_ok=True); docs.mkdir(parents=True,exist_ok=True)
    parity=optional(root,'parity.json'); diagnostic=optional(root,'diagnostic_summary.json'); selection=optional(root,'selection_seed17.json'); repeat=optional(root,'repeat_seed29.json'); prefix=optional(root,'prefix.json'); activation=optional(root,'activation.json'); evaluation=optional(root,'evaluation/store.json'); deps=optional(root,'dependency_manifest.json'); stages=optional(root,'stage_exits.json'); publication=optional(root,'publication/final.json')
    if execution_status is None: execution_status='COMPLETE' if all((parity,diagnostic,selection,repeat,prefix,activation,evaluation)) else 'BLOCKED_DEPENDENCY' if deps and deps.get('missing') else 'PARTIAL_ENGINEERING'
    science=science_status(diagnostic,activation,evaluation)
    publication_status='PUSH_VERIFIED' if publication and publication.get('status')=='PUSH_VERIFIED' else 'PUBLICATION_BLOCKED'
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
    registry=dict(status='LOCKED_CALIBRATION' if selection else 'NO_SELECTED_MODEL',BASE=BASE,parent_heads=spec['heads'],no_new_weights=True,lambdas={m:selection['selected'][m]['lambda_'] for m in METHODS} if selection else None,solver_schema=2,scientific_source_digest=source_digest())
    write(art/'selected_model_registry.json',registry)
    write(art/'resources.json',dict(actual_phase_resources=resources,new_optimizer_updates=0,new_rgb_encodings=0,new_sam_proposals=0,online_FPS=None,nested_timers_not_additive=True))
    # Published parent values are context, never observations from this task.
    parent_table=REPO/'artifacts/static_ovmap/frozen_fc_routing_v1/tables/table1_whole_map.json'
    inherited=read(parent_table); write(art/'inherited_FG_results.json',dict(status='INHERITED_PUBLISHED_CONTEXT_ONLY',BASE=BASE,source_path=str(parent_table.relative_to(REPO)),source_sha256=digest_file(parent_table),table=inherited,new_measurement=False))
    reason='; '.join(summary['missing']) or error or 'See formal phase records'
    table_links='\n'.join(f'- [Table {i}]({(Path("../../../")/ART/"tables"/name).as_posix()}.md)' for i,name in enumerate(('table1_baseline_checks','table2_diagnostics','table3_selection_repeat','table4_whole_map','table5_cost_scope'),1))
    # Links from this nested docs directory need four levels to repository root.
    table_links=table_links.replace('../../../artifacts/','../../../../artifacts/')
    atomic_text(docs/'RESULTS.md',f'''# FG2 surface bias wave1 v2 results

Execution: **{execution_status}**. Science: **{science}**. Deployment: N0_UNCHANGED.

{reason}

New optimizer updates / RGB encodings / SAM proposals: **0 / 0 / 0**. Missing experiment metrics remain null. Existing FG2 results are separately identified as inherited published context; they do not establish a gain for this new calibration. Neither offline cached replay nor these blocked phases establish online 30 FPS or an efficiency advantage over OVI-MAP.

{table_links}

All requested lambda candidates and the fixed-repeat status are retained. Seed29 and map labels never select lambda. CF18 contains18 captures from7 physical families and is not the complete ScanNet200 validation set. The parent proposals were derived from GT instances, so label-free bias fitting does not make the entire pipeline GT-free or training-free.

The gauge identifies relative offsets only. Equal view pooling with complete visibility can cancel the correction exactly; phi cannot manufacture a change. Independent-support nulls can degenerate. Spatial folds with bilinear-donor purge are not independent datasets because receptive fields may overlap.
''')
    atomic_text(docs/'CLAIMS.md','''# Evidence boundaries

Implemented hypothesis: estimate per-view relative raw-feature offsets from matched physical sites and intervene on values while keeping the frozen FG2 key/quality/query branch unchanged.

Allowed claims require measured evidence. A positive held-out residual statistic alone is neither statistical significance nor a causal demonstration. Map gains require both cohorts and both locked seeds; superiority over C_STAR requires its full map comparison. Fixed-owner semantic improvements do not establish better instance boundaries.

No claim of improved FG2 calibration, overall superiority to OVI-MAP, first-ever multi-view consistency, zero-shot training, independent full ScanNet200 validation, online30FPS, or new trained weights is supported by blocked experiments. C1/C2 gains without M1 superiority support routine calibration only.
''')
    atomic_text(docs/'HANDOFF.md',f'''# FG2 calibration handoff

BASE: `{BASE}`. Branch: `research/fg2-surface-bias-wave1-v2` (actual publication branch is in PUBLICATION_RECEIPT).
Execution: {execution_status}; science: {science}; deployment: N0_UNCHANGED.
Scientific source digest: `{source_digest()}`.

Original seed17 step1000 and seed29 step2000 heads are referenced by the [selected model registry](../../../../{ART}/selected_model_registry.json); no head was trained here. Lambda lock: {json.dumps(registry['lambdas'])}. The full parent source binding is verified before any in-memory path migration; parent_root keeps its inherited meaning. FG_ROOT is the CLI parent-root, LR_ROOT comes from that binding. Dependencies: see [dependency manifest](../../../../{ART}/dependency_manifest.json).

Use the existing controller environment and FC_python from the verified binding. Inherit one GPU string; the FC worker uses execution_config(binding)['gpu_lock'] and queued_gpu_lock. All writes belong to OUT; parent caches and upstream sources are read-only. Run from the worktree:

```bash
"$PY_CTRL" -u scripts/evaluation/run_fg2_surface_bias_wave1_v2.py --spec configs/static_ovmap/surface_bias_wave1_v2.json --parent-root "$FG_ROOT" --output-root "$OUT" --gpu "$GPU" --phase all --resume
PYTHONPATH=src "$PY_CTRL" -m pytest -q tests/test_surface_bias_wave1_v2.py
"$PY_CTRL" scripts/evaluation/run_fg2_surface_bias_wave1_v2.py --spec configs/static_ovmap/surface_bias_wave1_v2.json --parent-root "$FG_ROOT" --output-root "$OUT" --phase report --resume
```

Table Markdown/CSV/JSON come from sealed runtime JSON; the report phase regenerates them. Actual scientific worker exits are in stage_exits. The final all-process return code is observed by its invoking parent and stored externally, never predeclared in its own result commit.

Recovery point: restore the exact missing parent manifests, referenced grids, FC/text and input arrays; rerun identical spec/paths. Different scientific parameters require a new version. Do not recompute RGB, train, select another checkpoint, use similar caches or shrink cohorts. Blocked dependency: {reason}.

Results: [RESULTS](RESULTS.md). Limits and supported claims: [CLAIMS](CLAIMS.md). Requirement audit: [REVIEW](REVIEW.md). Large runtime arrays stay in OUT; Git contains compact tables and summaries. Cached replay performance is not online FPS.
''')
    checks=[('Frozen spec and isolated files','IMPLEMENTED'),('Relative Huber/KKT and strong simple controls','IMPLEMENTED_SYNTHETIC_TESTS'),('Seven-key raw-only adapter and same-seed fallback','IMPLEMENTED_PARENT_HEAD_SYNTHETIC_INPUTS'),('Real FC parity /69 original historical alignment','MEASURED' if parity else 'BLOCKED_NOT_VERIFIED'),('Donor-purged cross-fit diagnosis','MEASURED' if diagnostic else 'IMPLEMENTED_NOT_MEASURED'),('Full finite lambda selection / seed29 lock','MEASURED' if repeat else 'IMPLEMENTED_NOT_MEASURED'),('Prefix2/4 / condition isolation','MEASURED' if prefix else 'IMPLEMENTED_NOT_MEASURED'),('Fixed-owner original N/Q/F and official pools','MEASURED' if evaluation and evaluation.get('status')=='EVALUATION_COMPLETE' else 'IMPLEMENTED_NOT_VERIFIED_ON_REAL_MAP'),('Zero training/RGB/SAM','NO_NEW_SCIENTIFIC_RUNS' if not resources else 'ACTUAL_COUNTERS_RECORDED'),('Two normal publication commits','ACTUAL_PUBLICATION_PENDING_AT_C1_REPORT_SNAPSHOT')]
    atomic_text(docs/'REVIEW.md','# Primary-agent requirement audit\n\n'+''.join(f'- {name}: {status}.\n' for name,status in checks)+'\nThe full prompt was checked against implemented code and actual scope. Synthetic tests do not verify the real FC operator or restore missing dependencies. No missing numerical stage is marked complete; the new mechanism remains unassessed. Publication is proven only by actual C1/C2 remote SHA checks; see committed C1 receipt and external C2 final receipt.\n')
    inventory=[]
    for p in sorted(art.rglob('*')):
        if p.is_file() and p.name not in ('artifact_index.json','PUBLICATION_RECEIPT.json'): inventory.append(dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest_file(p)))
    write(art/'artifact_index.json',dict(scope='C1_TASK_ARTIFACTS_EXCLUDING_INDEX_AND_FUTURE_PUBLICATION_RECEIPT',scientific_source_digest=source_digest(),files=inventory,large_runtime_arrays='OUT_ONLY',regenerate='entrypoint --phase report with original runtime arguments'))
    return summary
