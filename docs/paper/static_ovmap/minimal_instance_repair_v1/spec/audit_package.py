from pathlib import Path
import ast, csv, hashlib, io, json, re, subprocess, sys, zipfile
P=Path(__file__).resolve().parent
spec=json.loads((P/'PROTOCOL_SPEC.json').read_text())
methods=[m['id'] for m in spec['methods']]
cohorts=spec['cohorts']; scenes=[s for v in cohorts.values() for s in v]
main=(P/'CODEX_FINAL_EXECUTION_EN.md').read_text()
contracts=(P/'IMPLEMENTATION_CONTRACTS.md').read_text()
ev=(P/'EVALUATION_AND_SELECTION.md').read_text()
zh=(P/'CODE_REVIEW_AND_PLAN_ZH.md').read_text()
source=(P/'SOURCE_EVIDENCE.md').read_text()
rows=[]
def check(name,flag,note=''):
    rows.append({'check':name,'passed':bool(flag),'note':note})
check('pin_and_target_branch',spec['base_commit']=='247e1e9782d43e882589bd9ab0015d513c200e49' and spec['branch']=='research/ovimap-minimal-instance-repair-v1')
check('nine_unique_methods',len(methods)==len(set(methods))==9)
check('26_unique_scenes',len(scenes)==len(set(scenes))==26)
check('8_and_18_cohorts',len(cohorts['replica8'])==8 and len(cohorts['scannet_cf18'])==18)
check('CF18_has_7_physical_families',len({s.split('_')[0] for s in cohorts['scannet_cf18']})==7)
check('234_scene_rows',len(methods)*len(scenes)==spec['scope']['scene_method_records']==234)
check('18_pools',len(methods)*len(cohorts)==spec['scope']['pooled_records']==18)
check('52_reused_182_new_max',2*len(scenes)==52 and spec['scope']['new_scene_records_max']==234-52)
check('zero_maps_frontend_training',all(spec['scope'][k]==0 for k in ['new_maps','new_segmentation_inference','new_native_NQ_inference','new_parameter_fits','new_backbones']))
check('pilot_scenes_in_cohorts',set(spec['pilots'])<=set(scenes) and spec['pilots']==['office1','scene0011_00'])
check('observer_832_bound',spec['observer']['max_frames']*len(scenes)==spec['scope']['full_evidence_rasterizations_max']==832)
check('unique_vs_failed_attempts_defined','SUCCESSFUL_UNIQUE' in spec['scope']['cap_count_policy'] and 'failed attempt/retry' in contracts)
check('eight_ops_four_units_one_incumbent',spec['support']['max_applied_operations']==8 and spec['support']['max_units_per_hypothesis']==4 and spec['support']['max_incumbents_per_hypothesis']==1)
check('old_core_and_raw0_are_frozen',not spec['support']['move_incumbent_rows'] and not spec['support']['modify_raw_zero_rows'] and not spec['support']['merge_two_incumbents'])
check('same_shared_library_and_no_cascade','SAME candidates' in main and 'single disjoint' in main and 'No validation-guided' in main)
check('mask_change_requires_new_evaluator_cache','partition-specific' in main and 'shared_masks' in contracts and 'ALL positive' in contracts)
check('single_partition_all_metrics',spec['evaluation']['single_exclusive_partition'] and 'same map and owner-to-class' in ev)
check('old_vs_new_geometry_identity_explicit','owner-array' in contracts and spec['evaluation']['base_geometry_unchanged'] and not spec['evaluation']['owner_partition_unchanged'])
check('loaded_official_ranking_and_pooling',spec['evaluation']['rank_mode']=='OFFICIAL_CURRENT_CLASS' and 'Do not use the mean of scene AP' in ev)
check('APall_nine_thresholds',len(spec['evaluation']['overlaps_apall'])==9 and spec['evaluation']['overlaps_apall'][0]==.5 and spec['evaluation']['overlaps_apall'][-1]==.9)
check('all_five_protected',spec['selection']['replica_all_five_nondecrease'] and spec['selection']['cf_all_five_nondecrease'] and len(spec['evaluation']['metrics'])==5)
check('simple_controls_allowed_to_win',set(spec['selection']['simplicity_order'])==set(methods[2:]) and 'Ordinary controls can win' in main)
check('fallback_and_deployment',spec['selection']['fallback']=='IR01_G1' and spec['selection']['deployment']=='N0_UNCHANGED')
check('no_claim_of_independent_confirmation','not independent data' in main and 'retrospective' in ev)
check('cold_64_bound',spec['timing']['methods_max']*spec['timing']['repeat_count']*len(cohorts['replica8'])==spec['scope']['cold_calls_max']==64)
check('cold_baseline_work_included','before G1 view/BVH construction' in ev and not spec['timing']['parent_recovery_cache_allowed'])
check('model_and_cache_identity_scope','model/text' in main and 'original incumbent support' in main)
check('pinned_existing_AnyUp',len(spec['anyup']['commit'])==40 and len(spec['anyup']['sha256'])==64 and spec['anyup']['checkpoint']=='anyup_paper.pth' and not spec['anyup']['use_natten'])
check('ordinary_AnyUp_not_geometry_attention',not spec['reread']['new_geometry_attention'] and 'disable' in contracts and 'owner/depth' in contracts)
check('no_extra_semantic_subset_by_GT','not GT or outcome' in main and 'do not choose the subset by this' in ev.lower())
check('maximum_support_units_and_views_declared',spec['reread']['max_incumbents']==16 and spec['reread']['max_views']==2 and spec['repair']['union_classification_views']==1)
check('observability_not_missing_file','present all-zero' in main and 'dependency block' in main)
check('consumed_not_entire_history_hashes','No full-repository test suite' in main and 'repeated global hashes' in main)
check('self_contained_command_exists_in_prompt','scripts/evaluation/run_ovimap_minimal_instance_repair.py' in main and '--phase all --resume' in main)
check('publication_and_receipt',all(x in main for x in ['MINIMAL_REPAIR_RESULTS.md','MINIMAL_REPAIR_HANDOFF.md','MINIMAL_REPAIR_SELECTION.md','MINIMAL_REPAIR_CLAIMS.md','git ls-remote','PUSH_VERIFIED','external publication/final.json']))
check('source_facts_distinguished_from_hypotheses','Proposed, not previously validated' in source and 'not mount the server' in source)
check('all_methods_in_main_and_matrix',all(m in main+(P/'EXPERIMENT_MATRIX.md').read_text() for m in methods))
for f in ['reference_kernels.py','test_reference_kernels.py']:
    try: ast.parse((P/f).read_text());ok=True
    except SyntaxError:ok=False
    check('syntax_'+f,ok)
log=(P/'REFERENCE_TEST_LOG.txt').read_text()
check('eight_reference_tests_actually_passed','Ran 8 tests' in log and log.rstrip().endswith('OK'))
missing=[]
for f in P.glob('*.md'):
    for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
        if not (link.startswith(('https://','http://','#','mailto:')) or (P/link).exists()): missing.append((f.name,link))
check('local_document_links_resolve',not missing,str(missing))
# Exact matrix export uses fixed method IDs; does not invent results.
sio=io.StringIO();cw=csv.writer(sio)
cw.writerow(['method_id','role','structure','semantics','cohort','required_scenes','status'])
for m in spec['methods']:
    for c,names in cohorts.items():
        cw.writerow([m['id'],m['role'],m['structure'],m['semantics'],c,len(names),'TO_EXECUTE_OR_IDENTITY_VERIFIED_PARENT_REUSE'])
(P/'EXPERIMENT_MATRIX.csv').write_text(sio.getvalue())
result={'scope':'INSTRUCTION_PACKAGE_ONLY_NOT_SERVER_EXPERIMENTS','base_commit':spec['base_commit'],
'checks':rows,'checks_passed':sum(x['passed'] for x in rows),'checks_total':len(rows),
'reference_tests':{'count':8,'status':'PASS','log':'REFERENCE_TEST_LOG.txt','uses_server_data':False},
'production_integration':'NOT_RUN','new_GPU_inference':0,'new_benchmark_results':0,'remote_repo_modified':False,
'manual_review_corrections':['Separated parent residual IoU diagnosis from full-raw geometry potential.',
'Explicitly allowed host support growth without transferring original incumbent core rows.',
'Partition content keys replace owner-ID-only evaluator cache reuse.',
'Zero observations are neutral; missing required files are not zero-valued evidence.',
'Science input ceilings apply to successful unique leaves; actual failed attempts/pilots/cold calls are separate.',
'Fixed barycentric tolerance and retained argmax weights to avoid implementation ambiguity.',
'Fixed one missing closing parenthesis in the illustrative reference kernel before the final passing run.'],
'overall_status':'PASS' if all(x['passed'] for x in rows) else 'FAIL'}
(P/'AUDIT_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['overall_status','checks_passed','checks_total']},indent=2))
for x in rows:
    if not x['passed']:print('FAIL',x)
if result['overall_status']!='PASS':sys.exit(1)
# A manifest excludes itself; final receipt is not recursively embedded in itself.
manifest={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(P.iterdir()) if f.is_file() and f.name!='FILE_MANIFEST.json'}
(P/'FILE_MANIFEST.json').write_text(json.dumps({'excludes':'FILE_MANIFEST.json','files':manifest},indent=2)+'\n')
Z=P.with_suffix('.zip')
with zipfile.ZipFile(Z,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(P.iterdir()):
        if f.is_file():z.write(f,P.name+'/'+f.name)
with zipfile.ZipFile(Z) as z:
    assert z.testzip() is None
    expected={P.name+'/'+f.name:f for f in P.iterdir() if f.is_file()}
    assert set(z.namelist())==set(expected)
    assert all(z.read(k)==f.read_bytes() for k,f in expected.items())
print('ZIP_VERIFIED',Z,len(expected),Z.stat().st_size)
