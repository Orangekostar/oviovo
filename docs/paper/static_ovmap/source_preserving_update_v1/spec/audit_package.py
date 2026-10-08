"""Fast specification audit only; no models, remote access, or parent data tests."""
from pathlib import Path
import hashlib
import json
import re
import ast
import subprocess
import sys
import datetime

ROOT=Path(__file__).resolve().parent
spec=json.loads((ROOT/'PROTOCOL_SPEC.json').read_text())
checks=[]
def check(name, result):
    checks.append({'name':name,'passed':bool(result)})
    if not result:
        raise AssertionError(name)

required=['README.md','CODEX_FINAL_EXECUTION_EN.md','IMPLEMENTATION_CONTRACTS.md',
          'EVALUATION_AND_SELECTION.md','CODE_REVIEW_AND_PLAN_ZH.md','EXPERIMENT_MATRIX.md',
          'SOURCE_EVIDENCE.md','PROTOCOL_SPEC.json','reference/decision_reference.py',
          'reference/test_decision_reference.py']
check('all_required_authored_files_present',all((ROOT/f).is_file() for f in required))
check('exact_base_commit',spec['base_commit']=='b000355eb8f91491e002df1499bd0170d91c35ef')
check('separate_target_branch',spec['branch']=='research/ovimap-source-preserving-update-v1')
check('separate_parent_and_output_roots',spec['parent_root']!=spec['output_root'] and not spec['output_root'].startswith(spec['parent_root']+'/'))
check('cohort_sizes',list(map(len,spec['cohorts'].values()))==[8,18])
check('distinct_26_scene_ids',len(set(sum(spec['cohorts'].values(),[])))==26)
check('cf18_seven_physical_families',len({s.split('_')[0] for s in spec['cohorts']['scannet_cf18']})==7)
ids=[m['id'] for m in spec['methods']]
check('nine_unique_methods',len(ids)==len(set(ids))==9)
check('exact_234_18_result_scope',spec['scope']['scene_method_records']==9*26 and spec['scope']['full_pools']==9*2)
check('52_inherited_182_matched_or_new',spec['scope']['inherited_main_rows']==52 and spec['scope']['new_or_identity_proven_main_rows']==182)
check('all_comparison_method_ids_exist',len(spec['comparisons'])==6 and all(a in ids and b in ids and a!=b for a,b in spec['comparisons']))
check('no_new_map_observer_view_frontend_NQ_AnyUp_or_fits',all(spec['scope'][n]==0 for n in ('new_maps','new_structural_operations','new_observer_rasterizations','new_view_searches','new_frontend_inference','new_NQ_inference','new_AnyUp_QK','new_backbones','parameter_fits')))
check('coarse_bound_matches_selection_and_views',spec['scope']['new_coarse_full_region_pools_upper_bound']==26*spec['parent_selection']['max_incumbents_per_scene']*spec['parent_selection']['max_full_views_per_incumbent']==832)
check('same_pair_domain_no_subset_shortcut',spec['pairing']['coarse_requires_all_parent_successful_full_views'] and not spec['pairing']['allow_result_based_owner_selection'])
check('mask_owner_and_recovery_preserved',spec['output']['owner_ids_bitwise_unchanged'] and spec['output']['all_G1_recovered_classes_unchanged'] and spec['output']['protected_raw_zero_policy']=='KEEP_WHOLE_INCUMBENT_CLASS_AS_PARENT')
check('fixed_alpha_eta_and_temperature',spec['operators']['alpha_F']==.5 and spec['operators']['eta_paired_delta']==.5 and spec['operators']['temperature_A']==spec['operators']['temperature_C']=='INHERIT_SAME_SCENE_T_F')
check('matched_A_mass_and_source_delta',spec['operators']['global_beta']=='alpha_F * original_dense_group_mass' and spec['operators']['residual_target']=='F_GROUP_ONLY' and 'same_view_C' in spec['operators']['residual'])
check('no_extra_acceptance_or_topk',spec['operators']['extra_acceptance_threshold'] is None and spec['operators']['residual_candidate_restriction']=='NONE_FULL_FROZEN_VOCABULARY')
check('nine_apall_thresholds_end_at_090',spec['evaluation']['APall_overlaps']==[.5,.55,.6,.65,.7,.75,.8,.85,.9])
check('all5_metrics_and_unrounded_gate',spec['evaluation']['metrics']==['apall','ap50','ap25','miou','macc'] and spec['selection']['epsilon_fraction']==1e-10 and spec['selection']['material_marker_cf_apall_delta_fraction']==.001)
check('all_new_and_matched_methods_can_win',set(spec['selection']['candidates'])==set(ids[2:])==set(spec['selection']['simplicity_order']))
check('no_gain_fallback_and_no_deployment',spec['selection']['fallback']=='SU01_G1' and spec['selection']['deployment']=='N0_UNCHANGED' and spec['selection']['independent_confirmation'] is False)
check('no_new_cold_measurement',spec['scope']['end_to_end_cold_calls']==0 and spec['timing']['mode']=='CONDITIONAL_CACHE_RESIDENT_CPU_DECISION_ONLY')
check('two_round_cpu_only_cost_design',spec['timing']['rounds']==2 and spec['timing']['samples_per_method']==8*2 and spec['timing']['batched_repeats_per_sample']==50)
check('bounded_tests_and_retry',spec['resources']['whole_repository_tests'] is False and spec['resources']['focused_test_functions_target']==10 and spec['resources']['identical_retry_max']==1)
main=(ROOT/'CODEX_FINAL_EXECUTION_EN.md').read_text()
check('all_methods_in_english_main',all(i in main for i in ids))
check('mandatory_all_resume_and_publish',all(k in main for k in ('--phase all --resume','git ls-remote','publication/final.json','source_preserving_update_v1.json')))
check('four_report_names_match',all(r in main for r in spec['publication']['four_reports']))
check('matched_not_historical_warning','matched replays, not automatically the historical' in main)
check('pair_residual_not_independence_claim','not a proven Bayesian likelihood-ratio' in main)
check('no_end_to_end_runtime_claim','There are zero newly authorized end-to-end cold calls' in main)
for f in ROOT.rglob('*.py'):
    ast.parse(f.read_text(),filename=str(f))
check('all_python_files_parse',True)
mds=list(ROOT.glob('*.md'))
check('all_markdown_fences_balanced',all(f.read_text().count('```')%2==0 for f in mds))
check('all_pinned_own_repo_links_at_base',all(spec['base_commit'] in url for f in mds for url in re.findall(r'https://github.com/Orangekostar/oviovo/blob/[^)\s]+',f.read_text())))
# One tiny synthetic run, not repeated production-style safety testing.
proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','reference','-p','test_*.py','-v'],cwd=ROOT,text=True,capture_output=True)
(ROOT/'REFERENCE_TEST_LOG.txt').write_text(proc.stdout+proc.stderr)
check('ten_reference_math_tests_pass',proc.returncode==0 and 'Ran 10 tests' in proc.stderr and '\nOK\n' in proc.stderr)
manual=[
 'Read-only exact-base parent binding; no recursive legacy controller execution.',
 'Same-view coarse C is distinct from historical F and uses the original v2 head/pooling.',
 'Common domain keeps selection/exclusions explicit; matched historical figures cannot be copied.',
 'Protected raw-zero rule stays identical; whole-owner semantics remain valid.',
 'No fine-feature or AnyUp regeneration is quietly counted as cached work.',
 'Global mixing A mass matches source-group mixing; missing groups have explicit behavior.',
 'Paired residual is applied to F only, uses same-view C and all classes.',
 'Any official rank changes are evaluated; recovery mask/class parity is not rank parity.',
 'Frozen formulas and both full cohorts; simple controls can win and no-gain still publishes.',
 'Timing is conditional CPU-only and not compared against prior standalone GPU times.',
 'Later training and acquisition research are explicitly out of scope, not claimed completed.',
 'Only the execution package and synthetic checks have been run here; no new scene results.'
]
report={'status':'PACKAGE_SPECIFICATION_AND_REFERENCE_MATH_AUDIT_PASSED',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'automated_checks':checks,'automated_check_count':len(checks),
        'reference_test_count':10,'reference_test_returncode':proc.returncode,
        'manual_design_review_items':manual,
        'real_server_data_checked':False,'real_FC_CUDA_executed':False,
        'new_scene_metrics_produced':False,'github_writes_performed':False,
        'limitations':'This audit checks consistency and small numerical properties; it is not production or scientific completion.'}
(ROOT/'AUDIT_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'reference_tests':10},ensure_ascii=False))
