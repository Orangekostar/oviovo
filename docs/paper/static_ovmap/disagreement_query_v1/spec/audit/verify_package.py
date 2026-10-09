"""Verify specification/document consistency; not production validation."""
from pathlib import Path
import ast, csv, json, re, hashlib, datetime
P=Path(__file__).resolve().parents[1]
s=json.loads((P/'PROTOCOL_SPEC.json').read_text())
checks=[]
def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})
    if not condition: print('FAILED',name)
with (P/'EXPERIMENT_MATRIX.csv').open() as f: rows=list(csv.DictReader(f))
a=[r for r in rows if r['stage']=='screen_A']; b=[r for r in rows if r['stage']=='screen_B']; full=[r for r in rows if r['stage']=='full']
texts={f.name:f.read_text() for f in P.glob('*.md')}
main=texts['CODEX_FINAL_EXECUTION_EN.md']; alg=texts['IMPLEMENTATION_CONTRACTS.md']; eva=texts['EVALUATION_AND_SELECTION.md']
check('pinned_base_SHA', bool(re.fullmatch('[0-9a-f]{40}',s['base_commit'])) and s['base_commit']=='9188e16d4ee7d0460de43413dfe73c225a400f41')
check('distinct_cohorts_8_18',list(map(len,s['cohorts'].values()))==[8,18] and len(set(sum(s['cohorts'].values(),[])))==26)
check('screen_is_fixed_4_subsets',sum(map(len,s['screen_cohorts'].values()))==4 and all(x in sum(s['cohorts'].values(),[]) for x in sum(s['screen_cohorts'].values(),[])))
check('screen_A_24_12',len(a)==6 and sum(int(r['scenes']) for r in a)==24)
check('screen_total_40_20',sum(int(r['scenes']) for r in a+b)==s['screen']['logical_rows']==40 and 2*len(a+b)==s['screen']['ordered_pools']==20)
check('full_156_12',sum(int(r['scenes']) for r in full)==s['full']['logical_rows']==156 and 2*len(full)==s['full']['ordered_pools']==12)
check('same_two_full_reads',s['views']['reads_per_policy']==2 and s['views']['same_anchor_all_policies'])
check('nonredundant_three_updates',s['update']['types']==['MEAN','AREA','SUPPORT'] and 'always reduces to the same mean' in main)
check('frozen_ordinary_FC_no_extra_models',all(s['scope'][k]==0 for k in ['new_maps','owner_partition_changes','new_segmentation','new_NQ','new_AnyUp','new_SAMV','new_training','new_text_encoding']))
check('new_evidence_mass_fixed',s['update']['new_evidence_mass']==.25 and s['update']['prior_mass']==.75 and '.75 * p0 + .25 * p_new' in alg)
check('candidate_future_scores_forbidden',s['disagreement']['candidate_cosines_before_selection']=='FORBIDDEN' and 'Only then acquire' in main)
check('GT_not_predicted_labels_forbidden',s['binding'].get('GT_labels_are_not_loaded_by_query_policy') and 'prediction_labels_are_not_loaded_by_query_policy' not in s['binding'])
check('physical_sites_bounded',s['observer']['max_sites_per_owner']==4096 and '4096 midpoint quantiles' in alg)
check('physical_not_epsilon_welding',s['observer']['surface_tolerance_welding_m']==0 and 'Do not epsilon-weld' in alg)
check('exact_pooler_fallback_reuse','area_fallback.region_vector' in alg and s['scope']['new_AnyUp']==0)
check('view_budget_not_FLOP_claim','equal FULL-image-read budget is not equal FLOPs' in main)
check('strict_full_gate_all_five',s['selection']['metrics']==['apall','ap50','ap25','miou','macc'] and s['selection']['both_cohorts_all_five_noninferior'])
check('strict_material_threshold_fraction',s['selection']['epsilon']==1e-10 and s['selection']['material_CF_apall_over_D2_fraction']==.001)
check('screen_gate_is_different_from_upgrade',s['screen']['extension']['candidate_vs_G1_apall_loss_budget_fraction']==.001 and not s['screen']['extension']['deployment_upgrade_from_screen'])
check('screen_gate_automatic_stop',s['screen']['extension']['if_none']=='STOP_AFTER_SCREEN_COMPLETE_AND_PUBLISH' and 'NOT_RUN_NO_SCREEN_SIGNAL' in eva)
check('no_historical_unmixing',not s['scope']['historical_evidence_unmixing'] and 'Historical N/Q/F remains the immutable p0 prior' in alg)
check('vocab_no_new_prompts_or_fake_AP',not s['diagnostics']['vocabulary_add_new_text'] and s['diagnostics']['reduced_vocab_AP']=='DO_NOT_COMPUTE')
check('screen_budget_arithmetic',64*(5+4)==s['resources']['screen_pool_head_max']==576 and 4*32==s['resources']['screen_unique_FC_images_max'])
check('full_budget_arithmetic',22*16*7==s['resources']['additional_22_full_pool_head_max']==2464 and 576+2464==s['resources']['science_cumulative_pool_head_max'])
check('cold_calls_conditional_32',8*2*2==s['resources']['cold_calls_max'] and s['timing']['trigger']=='FULL_STRICT_TARGET_PASS_ONLY')
check('cold_budget_bounds',32*32==s['resources']['cold_unique_FC_image_inputs_upper_bound'] and 32*16*6==s['resources']['cold_pool_head_upper_bound'])
check('original_cohort_order_preserved',s['cohorts']['replica8'][0]=='office0' and s['cohorts']['scannet_cf18'][-1]=='scene0518_00')
check('new_branch_and_external_publication',s['branch'] in main and s['publication']['external_receipt']=='publication/final.json' and 'git ls-remote' in main)
check('real_cli_required','--phase all --resume' in main and 'actual successful complete CLI' in texts['README.md'])
check('bounded_tests_not_full_suite',s['testing']['production_directed_tests_max']==12 and not s['testing']['full_repository_suite'] and 'thousands of duplicated requirement assertions' in main)
check('source_blob_registry_present',len(re.findall(r'^S\d{2} [a-f0-9]{40}$',texts['SOURCE_EVIDENCE.md'],flags=re.M))==12)
check('no_TODO_or_todo_placeholder',not any(re.search(r'\bTODO\b|\bTBD\b|PLACEHOLDER_RESULT',x) for x in texts.values()))
for p in (P/'reference').glob('*.py'): ast.parse(p.read_text(),filename=str(p))
check('reference_Python_syntax',True)
log=(P/'audit/REFERENCE_TEST_LOG.txt').read_text()
check('reference_12_tests_passed',bool(re.search(r'Ran 12 tests',log)) and log.rstrip().endswith('OK'))
# Every internal markdown file link that is actually used must resolve.
bad=[]
for name,text in texts.items():
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if not target.startswith(('http:','https:','#','mailto:')):
            if not (P/target.split('#')[0]).exists(): bad.append((name,target))
check('internal_markdown_links_resolve',not bad)
report={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'check_count':len(checks),'passed':sum(x['passed'] for x in checks),'checks':checks,'reference_tests':12,'not_executed':['production FC','Open3D direct visibility','full dataset/scorer replay','new GPU experiments','GitHub write or push'],'scope':'Instruction consistency and tiny numerical reference contracts only'}
(P/'audit/AUDIT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','check_count','passed','reference_tests')},ensure_ascii=False))
raise SystemExit(0 if report['status']=='PASS' else 1)
