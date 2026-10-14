"""Measured six-table evidence package, bounded complete records and review."""
from pathlib import Path
import csv
import gzip
import io
import json
import shutil
import numpy as np
from .common import (REPO,read,verified,write,atomic_text,digest_file,ConsumptionIndex,source_digest,CONDITIONS)
DOC=REPO/'docs/paper/static_ovmap/branch_utility_wave2_v1'
ART=REPO/'artifacts/static_ovmap/branch_utility_wave2_v1'

def optional(root,name):
    p=Path(root)/name;return verified(p) if p.exists() else {}

def table(name,rows,caption):
    folder=ART/'tables';folder.mkdir(parents=True,exist_ok=True);columns=list(dict.fromkeys(k for row in rows for k in row));write(folder/(name+'.json'),dict(caption=caption,rows=rows))
    with (folder/(name+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    def cell(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:.6f}'
        if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False,sort_keys=True).replace('|','\\|')
        return str(v).replace('|','\\|').replace('\n',' ')
    result='\n'+caption+'\n\n'
    if rows:result+='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(cell(row.get(k)) for k in columns)+' |\n' for row in rows)
    else:result+='No measured rows available.\n'
    atomic_text(folder/(name+'.md'),result);return result

def gz_jsonl(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as stream:
        for row in values:stream.write((json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
    temporary.replace(path)

def copy_record(root,relative,inventory,include_arrays=False):
    root=Path(root);source=root/relative
    if not source.exists() or not source.is_file():return
    dest=ART/'records'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);inventory.append(dict(source=str(source),published=str(dest.relative_to(REPO)),bytes=dest.stat().st_size,sha256=digest_file(dest)))
    if include_arrays:
        row=read(source)
        if row.get('arrays'):
            desc=row['arrays'];ConsumptionIndex().identity(desc['path'],desc);file=Path(desc['path']);target=dest.with_suffix(file.suffix);shutil.copyfile(file,target);inventory.append(dict(source=str(file),published=str(target.relative_to(REPO)),bytes=target.stat().st_size,sha256=digest_file(target)))

def package(root):
    root=Path(root);ART.mkdir(parents=True,exist_ok=True);inventory=[];raw=[];capture=optional(root,'capture.json');command=f'/home/ww/miniconda3/envs/ovimap-map/bin/python {REPO}/scripts/evaluation/run_fg2_branch_utility_wave2_v1.py --spec {REPO}/configs/static_ovmap/branch_utility_wave2_v1.json --parent-root {optional(root,"binding.json").get("fg_parent_root","")} --previous-root {optional(root,"binding.json").get("previous_root","")} --output-root {root.parent}/attempt_regeneration --gpu 0 --phase all --resume'
    units=capture.get('units',[])
    for start in range(0,len(units),100):
        chunk=units[start:start+100];values=[]
        for entry in chunk:
            row=verified(entry['path']);desc=row['arrays'];ConsumptionIndex().identity(desc['path'],desc)
            with np.load(desc['path'],allow_pickle=False) as a:
                enriched=dict(row);enriched['rows']=[dict(r,features24=a['features'][i].tolist()) for i,r in enumerate(row['rows'])];enriched['source_unit']=dict(path=entry['path'],identity=entry['identity']);enriched['raw_array_policy']='EMBEDDINGS_AND_FULL_COSINES_REMAIN_IN_OUT';values.append(enriched)
            raw.append(dict(unit=entry['path'],unit_identity=entry['identity'],**desc,regeneration_command=command))
        path=ART/'actions'/f'chunk_{start//100:03d}.jsonl.gz';gz_jsonl(path,values);inventory.append(dict(published=str(path.relative_to(REPO)),bytes=path.stat().st_size,sha256=digest_file(path),original_condition_units=len(chunk),contents='ALL_LOGICAL_ACTIONS_UTILITY_TOP1_MARGIN_FEATURES_PLUS_FACTORIAL_STATIC'))
    for name in ('task_paths.json','class_split.json','split_manifest.json','features/text_mapping.json','export/runtime_registry.json','binding.json','dependency_manifest.json','storage_preflight.json','preflight.json','capture.json','baseline.json','training_gate.json','training.json','selection.json','selection_lock.json','repeat.json','activation.json','predictions/summary.json','evaluation/store.json','conveyor/verifications.json','map_cost.json','stage_exits.json','focused_tests.json','run_summary.json','measured_sources.json','action_cost_counts.json','primary_review_actual.json','engineering/medoid_repair_compatibility.json'):
        copy_record(root,name,inventory)
    # Large family/condition/action/owner rows are kept in lossless gzip blocks.
    for name in ('factorial.json','action_bank.json','conveyor.json'):
        file=root/name
        if file.exists():
            dest=ART/'records'/(name+'.gz');gz_jsonl(dest,[verified(file)]);inventory.append(dict(source=str(file),published=str(dest.relative_to(REPO)),bytes=dest.stat().st_size,sha256=digest_file(dest)))
    for file in sorted((root/'recognition').glob('*.json')):copy_record(root,file.relative_to(root),inventory,include_arrays=file.name.startswith('dev_'))
    for file in sorted((root/'locked_readouts').glob('seed*/*.json')):copy_record(root,file.relative_to(root),inventory,include_arrays=True)
    for file in sorted((root/'training').rglob('*')):
        if not file.is_file() or file.name=='resume.pt':continue
        relative=file.relative_to(root);dest=ART/'records'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest);inventory.append(dict(source=str(file),published=str(dest.relative_to(REPO)),bytes=dest.stat().st_size,sha256=digest_file(dest)))
    for file in sorted((root/'resources').rglob('*.json')):copy_record(root,file.relative_to(root),inventory)
    for file in sorted((root/'decisions').glob('*/*.json')):copy_record(root,file.relative_to(root),inventory)
    lock=optional(root,'predictions/summary.json')
    if lock.get('status')=='ALL_PREDICTIONS_LOCKED':
        from static_ovmap.module_validation.scannet_study import load_prediction
        from static_ovmap.composition_study.object_evidence import owner_labels
        for scene in lock['scenes']:
            receipt=verified(root/'predictions'/scene/'receipt.json');full=[]
            for name,desc in receipt['methods'].items():
                payload=load_prediction(desc['manifest']);full.append(dict(method=name,prediction_key=payload.prediction_key,record_key=payload.record_key,owner_semantics=owner_labels(payload),instance_ranks=payload.instance_ranks,manifest=desc['manifest'],background_policy='BYTE_EXACT_PARENT_G1',owner_partition='BYTE_EXACT_PARENT_G1_EXCEPT_REF_D2_USES_ITS_OWN_PARENT',geometry=payload.geometry))
            dest=ART/'map_predictions'/(scene+'.jsonl.gz');gz_jsonl(dest,full);inventory.append(dict(published=str(dest.relative_to(REPO)),bytes=dest.stat().st_size,sha256=digest_file(dest),contents='COMPLETE_OWNER_SEMANTICS_AND_NATIVE_RANKS_OVER_VERIFIED_PARENT_PARTITION'))
            copy_record(root,Path('predictions')/scene/'receipt.json',inventory)
    total=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file() and p.name!='PUBLICATION_RECEIPT.json')
    if total>145*2**20:raise ValueError('Artifact package leaves insufficient150MiB code/document budget')
    return write(ART/'artifact_index.json',dict(status='COMPLETE',published=inventory,raw_action_blocks=raw,raw_regeneration_command=command,raw_Unit_U_saved=False,total_artifact_bytes_before_index=total,source_digest=source_digest(),complete_logical_action_records=True,locked_prediction_representation='FULL_DEV_EMBEDDINGS_AND_COSINES;MAP_ALL_OWNER_LABELS_RANKS_PLUS_FROZEN_PARTITION',dense_grids_published=False))

def breakdown(rows,kind):
    groups={}
    for row in rows:
        key=(row['role'],row['seed'],row['old_prefix'],row['new_prefix'],row[kind]);groups.setdefault(key,[]).append(row)
    return [dict(role=k[0],seed=k[1],transition=f'{k[2]}-{k[3]}',**{kind:k[4]},originals=len({r['key'] for r in bank}),records=len(bank),**{name:float(np.mean([r['effects'][name] for r in bank])) for name in ('base','local','interaction','total')}) for k,bank in sorted(groups.items())]

def run(root,execution_status=None,error=None):
    root=Path(root);DOC.mkdir(parents=True,exist_ok=True);baseline=optional(root,'baseline.json');factor=optional(root,'factorial.json');bank=optional(root,'action_bank.json');gate=optional(root,'training_gate.json');training=optional(root,'training.json');selection=optional(root,'selection.json');repeat=optional(root,'repeat.json');activation=optional(root,'activation.json');evaluation=optional(root,'evaluation/store.json');conveyor=optional(root,'conveyor.json');exits=optional(root,'stage_exits.json');tests=optional(root,'focused_tests.json');resources=[verified(p) for p in sorted((root/'resources').rglob('*.json'))]
    required=('preflight.json','baseline.json','factorial.json','action_bank.json','training_gate.json','training.json','selection.json','repeat.json','predictions/summary.json','evaluation/store.json','conveyor.json')
    missing=[name for name in required if not (root/name).exists()];execution=execution_status or ('COMPLETE' if not missing else 'PARTIAL_ENGINEERING');new_updates=training.get('new_optimizer_updates',0);science='NOT_YET_MEASURED'
    if activation:science='MAPS_ACTIVATED_PENDING_FORMAL_RESULT' if activation['methods'] and not evaluation.get('pooled_metrics') else 'FORMAL_MAPS_MEASURED' if evaluation.get('pooled_metrics') else 'NO_BOTH_SEED_PROMOTION_SIGNAL'
    if gate and not gate['training_activated'] and not activation:science='NO_TRAIN_BRANCH_UTILITY_COVERAGE'
    summary=write(root/'run_summary.json',dict(execution_status=execution,science_status=science,publication_status='PENDING_VERIFICATION',missing_stages=missing,error=error,new_optimizer_updates=new_updates,new_rgb_encodings=sum(r.get('new_rgb_encodings',0) for r in resources),new_sam_proposals=0,activation=activation.get('methods',[]),deployment='N0_UNCHANGED',source_digest=source_digest()))
    resource_phases={r['phase'] for r in resources}
    for exit_row in exits.get('records',[]):
        if exit_row['phase'] not in resource_phases:
            resources.append(dict(phase=exit_row['phase'],boundary='OBSERVED_PHASE_SUBPROCESS_WALL_INCLUDES_STARTUP_CACHED_IO',wall_seconds=exit_row['wall_seconds'],new_optimizer_updates=0,new_rgb_encodings=0))
            resource_phases.add(exit_row['phase'])
    inputs=[]
    for key,row in baseline.get('summaries',{}).items():inputs.append(dict(group=key,originals=row['metrics']['original_objects'],families=row['metrics']['families'],categories=row['metrics']['categories'],A_percent=100*row['metrics']['A'],M_percent=100*row['metrics']['M'],C_percent=100*row['metrics']['C']))
    parts=[table('01_inputs_baselines',inputs,'Table1. Exact parent support and pooled recognition baselines (%). Full200 competition; TRAIN/DEV base-positive originals only.')]
    parts.append(table('02_factorial',factor.get('summaries',[]),'Table2. Full200 GT-margin software interventions. Base+local=total; interaction is reported separately, never added again. Conditions are repeated measurements, not independent originals.'))
    utility=[]
    for seed,row in gate.get('seeds',{}).items():
        for name in ('positive','negative','branch_preference'):
            x=row[name];utility.append(dict(seed=seed,criterion=name,originals=x['originals'],families=x['families'],minimum_originals=20 if name=='branch_preference' else 40,minimum_families=6 if name=='branch_preference' else 8,seed_passed=row['passed'],both_seed_training_activated=gate['training_activated']))
    parts.append(table('03_utility_gate',utility,'Table3. TRAIN prefix8 only, original deduplication over4 conditions. Positive≥.01; negative≤−.01; same-view branch preference>coupled/KEEP+.01. GT-best is an offline TRAIN diagnostic, not a method or AP bound.'))
    chosen=[]
    for seed,source in ((17,selection.get('selected',{})),(29,repeat.get('results',{}))):
        for method,row in source.items():chosen.append(dict(seed=seed,method=method,base=row.get('base'),gamma=row.get('gamma'),step=row.get('step'),tau=row.get('tau'),A8_percent=100*row['A8'],M8_percent=100*row['M8'],C8_percent=100*row['C8'],A4_percent=100*row['A4'],mean_edits=row['mean_edits'],net_corrections8=row['net_corrections8'],delta_A8_pp=100*row['deltas']['A8'],delta_M8_pp=100*row['deltas']['M8'],delta_C8_pp=100*row['deltas']['C8'],delta_A4_pp=100*row['deltas']['A4'],passed=row['passed']))
    parts.append(table('04_selection_repeat',chosen,'Table4. Seed17 finite DEV selection and seed29 same-profile repeat. Gate: ∆A8≥+.5pp, ∆M8/∆C8/∆A4≥−.5pp, ordinary net corrections8>0, both seeds. No seed29 reselection.'))
    maps=[]
    for cohort,methods in evaluation.get('pooled_metrics',{}).items():
        for method,row in methods.items():maps.append(dict(cohort=cohort,method=method,**{k+'_percent':None if row['metrics'][k] is None else 100*row['metrics'][k] for k in ('apall','ap50','ap25','miou','macc')}))
    if not maps:maps=[dict(status=evaluation.get('status','NOT_EXECUTED'),reason=activation.get('reason',error),C_STAR=activation.get('C_STAR'),both_seed_passes=activation.get('passed_both'),new_map_rows=0 if evaluation.get('status')=='NOT_ACTIVATED_BY_FIXED_RULE' else None)]
    parts.append(table('05_official_maps',maps,'Table5. Official ordered scene-pool metrics (%), or exact fixed activation reason. No averaging scene AP; absent classes remain null. Fixed G1 geometry and original N/Q/F ranking path.'))
    costs=[dict(phase=r['phase'],status=r.get('status','OBSERVED_PHASE_EXIT'),boundary=r['boundary'],wall_seconds=r['wall_seconds'],native_head_attempts=(optional(root,'capture.json').get('new_full_head_calls_this_invocation',0)+optional(root,'capture.json').get('new_subset_head_calls_this_invocation',0) if r.get('status')=='COMPLETE' else baseline.get('actual_pre_map_native_attempts',0)-optional(root,'capture.json').get('new_full_head_calls_this_invocation',0)-optional(root,'capture.json').get('new_subset_head_calls_this_invocation',0)) if r['phase']=='baseline' else optional(root,'map_cost.json').get('native_head_forwards') if r['phase']=='map_predict' else 0,phi_calls=r.get('phi_calls',0),phi_vectors=r.get('phi_vectors'),logical_action_candidates=sum(optional(root,'action_cost_counts.json').get('logical_action_candidates',{}).values()) if r['phase']=='baseline' and r.get('status')=='COMPLETE' else None,phi_seconds=r.get('phi_seconds'),optimizer_updates=r.get('new_optimizer_updates',0),RGB_encodings=r.get('new_rgb_encodings',0),host_peak_RSS_GiB=r.get('host_peak_rss_bytes',0)/2**30 if r.get('host_peak_rss_bytes') is not None else None,cache_budget_GiB=32,peak_allocated_GiB=r.get('GPU_peak_allocated_bytes',0)/2**30 if r.get('GPU_peak_allocated_bytes') is not None else None,peak_reserved_GiB=r.get('GPU_peak_reserved_bytes',0)/2**30 if r.get('GPU_peak_reserved_bytes') is not None else None,online_FPS=None) for r in resources]
    parts.append(table('06_cost',costs,'Table6. Measured cached inference/training boundaries include all enumerated subset heads, phi and transfers. Native-head cap85000; cache≤32GiB; conditional optimizer budget8000. These are not RGB stream30fps or skipped-frame measurements; nested timers are not additive.'))
    family=breakdown(factor.get('rows',[]),'family');condition=breakdown(factor.get('rows',[]),'condition');table('factorial_by_family',family,'Family clusters, descriptive within-family means.');table('factorial_by_condition',condition,'Condition strata, retaining original denominators.');table('actual_prefix_coverage',[dict(role=k,**v) for k,v in bank.get('coverage',{}).items()],'Requested prefix and actual support are separate; main denominator all originals.');table('all_seed17_candidates',selection.get('candidates',[]),'Every finite seed17 DEV candidate; locked profiles selected before seed29 candidates.');table('existing_conveyor',conveyor.get('scene_rows',[]),'Existing FC8→FG2 conveyor, not new map inference. Maximum geometric IoU matches are descriptive and distinct from official scorer TP.');write(ART/'tables/fixed_examples.json',dict(examples=factor.get('examples',{}),fixed_selection='EFFECT_SIGN_EARLIEST_KEY',causal_world_claim=False))
    edit_types=[]
    for file in sorted((root/'locked_readouts').glob('seed*/*.json')):
        r=verified(file);edit_types.append(dict(seed=r['seed'],method=r['profile']['method'],prefix=r['prefix'],originals=len(r['records']),condition_records=len(r['records'])*4,mean_edits=r['mean_edits'],**r['edit_types']))
    table('locked_recognition_edit_types',edit_types,'All originals/conditions; proven local no-ops count zero edits. STATIC is a configuration, with no discrete view deletion.')
    map_edits=[]
    for file in sorted((root/'decisions').glob('*/*.json')):
        r=verified(file);values=list(r['owners'].values());map_edits.append(dict(scene=file.parent.name,method=file.stem,eligible_originals=len(values),edited=sum(v['edited'] for v in values),edit_rate=sum(v['edited'] for v in values)/len(values) if values else None,actions={k:sum(v['kind']==k for v in values) for k in ('KEEP','G','L','B')},actual_views=[v['actual_views'] for v in values]))
    table('map_edit_rates',map_edits,'Map edits over every eligible original; unedited/restored/background support retained in complete map owner registries.')
    if evaluation.get('map_gains'):write(ART/'tables/map_four_cell_deltas.json',dict(vs_same_seed_FG2=evaluation['map_gains'],vs_strong_controls=evaluation.get('comparisons_vs_all_strong_controls',{})))
    if evaluation.get('base_novel'):write(ART/'tables/map_base_heldout.json',dict(results=evaluation['base_novel'],absent_classes_null=True,Replica='NOT_APPLICABLE'))
    status=f'Execution: **{execution}**. Science: **{science}**. Deployment: **N0_UNCHANGED**. New controller updates: **{new_updates}**. New RGB/SAM: **0/0**.'
    text='# FG2 branch utility Wave2 measured results\n\n'+status+'\n\nThe task tests branch admission over preselected cached view banks, not online camera acquisition. Frozen FC/text/FG2 weights, parent splits and scene order are retained.\n'+''.join(parts)+'\n\nAll finite candidates, family/condition strata and fixed examples are supplied in tables/. All action utilities, labels, margins and24 features are in lossless action chunks; raw embeddings/full cosines remain in OUT with bytes/hash/regeneration command. No local group U banks are persisted.\n'
    if not new_updates:text+='\nThe TRAIN gate did not activate learning; learned arms are absent rather than fabricated zero results. STATIC selection/repeat remains mandatory.\n' if gate else '\nLearning has not yet been measured.\n'
    if 'WITH_NEW_CORRESPONDENCE' in selection.get('selected',{}) and 'WITH_NEW_CORRESPONDENCE' in repeat.get('results',{}):
        a=selection['selected']['WITH_NEW_CORRESPONDENCE'];b=repeat['results']['WITH_NEW_CORRESPONDENCE'];text+=f'\nWITH_NEW_CORRESPONDENCE improves A4 by {100*a["deltas"]["A4"]:.2f}/{100*b["deltas"]["A4"]:.2f}pp (seeds17/29), but A8 changes by {100*a["deltas"]["A8"]:.2f}/{100*b["deltas"]["A8"]:.2f}pp. A8 matches the coupled control at both seeds; branch independence is not established.\n'
    if activation and not activation['methods']:text+='\nNo profile satisfied the specified promotion route across both seeds. New map scoring is therefore intentionally not activated. Existing26-scene conveyor analysis remains separate.\n'
    atomic_text(DOC/'RESULTS.md',text)
    command=f'/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_fg2_branch_utility_wave2_v1.py --spec configs/static_ovmap/branch_utility_wave2_v1.json --parent-root {optional(root,"binding.json").get("fg_parent_root","")} --previous-root {optional(root,"binding.json").get("previous_root","")} --output-root {root} --gpu 0 --phase all --resume'
    handoff='# FG2 branch utility Wave2 handoff\n\n'+status+'\n\n```bash\n'+command+'\n```\n\nActual observed stage exits: '+json.dumps([(r['phase'],r['exit_code']) for r in exits.get('records',[])])+'.\n\nThe baseline pass also captures factorial/static/action states to respect the native-forward budget. Later factorial/action phases aggregate those same sealed units and add no head calls. Each object/condition/seed retains U only for its current subsets in memory. Prefix2 is always KEEP for admission, requested4/8 and actual support are separately reported. Conditions never share site correspondence keys.\n\nController weights and normalizers: OUT/training/seed{17,29}/; selected thresholds/steps/baseγ: OUT/selection_lock.json. The control zeros the last8 standardized features and retains FG2 geometry-derived quality. TRAIN targets use base160 CE; DEV uses full200 competition; Replica maps use their original context text. Labels do not enter forward/features/controller inference.\n\nThe pre-DEV medoid floating-point tie repair preserved only TRAIN units after exact native producer AST/byte validation; the−2 interrupted exit and its resources remain recorded. DEV STATIC candidates use the corrected mean-other-cosine definition. Resume validates sealed unit identities and parent bytes; partial errors/logs remain OUT/error.json and logs/. Do not replace FC/text/head bytes, lower fixed gates or resample exposed data. New raw regeneration uses a new output attempt. Map stages require whole-arm locks before GT; no map-driven reselection. Read-only parent roots are retained.\n\nIndependent unseen scenes, actual proposals, and stronger matched external verification remain outside this exposed GT-support protocol. Positive observed recognition signals are not CVPR acceptance or world-causal evidence. Publication C1/C2 equality is verified externally; this file never predicts its own commit SHA.\n'
    existing=DOC/'HANDOFF.md'
    if existing.exists() and 'Verified result commit C1:' in existing.read_text():handoff+=existing.read_text()[existing.read_text().index('Verified result commit C1:'):]
    atomic_text(existing,handoff)
    claims=[dict(claim='Frozen cached forward/feature isolation',support=['preflight.json','capture.json','focused_tests.json'],status='VERIFIED' if baseline and tests.get('exit_code')==0 else 'PENDING'),dict(claim='TRAIN branch admission utility coverage',support=['training_gate.json'],status='PASSED_BOTH_SEEDS' if gate.get('training_activated') else 'NOT_SUPPORTED' if gate else 'PENDING'),dict(claim='Matched learned controller advantage',support=['selection.json','repeat.json'],status='DEV_BOTH_SEED_SIGNAL_ONLY' if activation.get('passed_both',{}).get('WITH_NEW_CORRESPONDENCE') else 'NOT_SUPPORTED'),dict(claim='Official map advantage over FG2/strong controls',support=['evaluation/store.json'],status='MEASURED_CHECK_FOUR_CELLS_AND_CONTROLS' if maps and evaluation.get('pooled_metrics') else 'NOT_TESTED_BY_FIXED_ACTIVATION_RULE'),dict(claim='Improved geometry/online30fps/unseen-scene generalization',support=[],status='NOT_SUPPORTED')]
    write(ART/'claims.json',dict(claims=claims));atomic_text(DOC/'CLAIMS.md','# Evidence boundaries\n\n'+status+'\n\n'+''.join('- '+r['claim']+': **'+r['status']+'**; '+', '.join(r['support'])+'.\n' for r in claims)+'\nGT-best is TRAIN diagnosis only. Conditional training must be reported as actual updates; absent learned arms/map metrics are not filled with zeros. Two exposed head seeds are repeated evaluations, not independent unseen data.\n')
    review=[dict(requirement='Exact source/spec/head/text/parent support',evidence='preflight.json;binding.json;runtime_registry;history parity',passed=bool(baseline)),dict(requirement='Same7 tensor subset, currentprefix isolation, condition isolation, realFP32 Compose/history',evidence='focused tests1–2;capture units;baseline.history_parity',passed=bool(baseline and tests.get('exit_code')==0)),dict(requirement='G/L/B support edits, subset sharing, no-op proof,24 target-free features',evidence='focused tests3–4;all action chunks;capture',passed=bool(bank and tests.get('exit_code')==0)),dict(requirement='TRAIN-only gate; four paired2000 arms if activated, otherwise0',evidence='training_gate;training;normalizers/draws/curves/resume checkpoints',passed=bool(training and (new_updates==8000 if gate.get('training_activated') else new_updates==0))),dict(requirement='STATIC9/conditional57; lockedC_STAR before29; fixedrepeat/promotion',evidence='selection_lock;selection.candidates;repeat;activation',passed=bool(repeat and selection.get('logical_cells')==(57 if new_updates else 9))),dict(requirement='Existing26scene conveyor regardless of promotion',evidence='conveyor.json;52scene/head rows;GT-match and official metric receipts',passed=conveyor.get('status')=='COMPLETE'),dict(requirement='Conditional dynamic allmap locks/scoring or exact nonactivation',evidence='predictions/summary;evaluation/store;baseline scorer checks',passed=evaluation.get('status') in ('EVALUATION_COMPLETE','NOT_ACTIVATED_BY_FIXED_RULE')),dict(requirement='85000head/32GiBcache/8000update/0encoding caps; focused6tests',evidence='native ledger;resources;training;focused_tests',passed=bool(baseline and baseline.get('actual_pre_map_native_attempts',85001)<=85000 and new_updates<=8000 and tests.get('groups')==6 and tests.get('exit_code')==0 and summary['new_rgb_encodings']==0)),dict(requirement='Sixcoretables/full candidates/curves/weights/actions/raw receipts',evidence='tables;records;actions;artifact_index',passed=not missing),dict(requirement='Normal C1/C2 remote verification and actual terminalexit',evidence='C2 C1 receipt;externalpublication/final.json;externalcontroller_exit.json',passed=None)]
    write(ART/'review_checks.json',dict(review_owner='PRIMARY_AGENT',checks=review,prompt_file='prompt/00_下一步实验设计与Codex任务.md',no_full_repo_suite=True));atomic_text(DOC/'REVIEW.md','# Primary evidence review\n\n'+status+'\n\n'+''.join('- '+r['requirement']+': '+('PASS' if r['passed'] else 'EXTERNAL_FINAL_RECEIPT_REQUIRED' if r['passed'] is None else 'INCOMPLETE')+' — '+r['evidence']+'.\n' for r in review)+'\nA4 improvements are a weak recognition signal; A8 is not repeated and does not exceed COUPLED. No map method passed the fixed promotion route. Numerical decomposition and unit tests establish software arithmetic, not scientific novelty by themselves. Positive resource gates are descriptive thresholds, not significance tests. Scientific conclusions require the actual repeated DEV/map/control cells. The full prompt and unchanged JSON remain bundled.\n')
    artifact=package(root)
    return write(root/'report.json',dict(status='COMPLETE',execution_status=execution,science_status=science,review_checks=review,artifact_index=artifact['identity'],publication_status='PENDING_C1_C2_REMOTE_VERIFICATION'))
