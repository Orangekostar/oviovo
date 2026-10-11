"""Finite package consistency audit; not a server validation suite."""
from pathlib import Path
import json, csv, re, ast, hashlib
P=Path(__file__).resolve().parents[1]
s=json.loads((P/'PROTOCOL_SPEC.json').read_text())
checks=[]
def check(name,predicate,detail=''):
 checks.append(dict(check=name,passed=bool(predicate),detail=detail))
 if not predicate:raise AssertionError(name)
required=['README.md','CODEX_FINAL_EXECUTION_EN.md','CODE_REVIEW_AND_PLAN_ZH.md','MODEL_AND_TRAINING_CONTRACT.md','DATA_AND_BINDING.md','EVALUATION_SELECTION_RELEASE.md','DIAGNOSTICS_CONTRACT.md','PROTOCOL_SPEC.json','EXPERIMENT_MATRIX.csv','SOURCE_EVIDENCE.md','reference_math.py','tests/test_reference_math.py']
check('required_files_present',all((P/f).is_file() for f in required))
check('anchor_and_branch',s['base_commit']=='76db115aa7df652a0b2dc4575df52e6a74a63549' and s['branch']=='research/ovimap-frozen-fc-routing-v1')
check('R_prerequisite_removed',s['lineage']['no_required_R_qualification'] and not s['lineage']['old_task_binders_and_trainers_called'])
methods=s['methods'];check('exact_four_arms',[m['id'] for m in methods]==['FG1_VIEW','FG2_SURFACE','FG3_AUX','FG4_ROUTE'])
check('matched_pair_axes',methods[0]['grouping']=='VIEW' and all(m['grouping']=='SURFACE' for m in methods[1:]) and [m['membership_loss'] for m in methods]==[False,False,True,True] and [m['direct_routing'] for m in methods]==[False,False,False,True])
check('no_MA_or_free_value_transform',not s['model']['trainable_MaskAdapter'] and not s['model']['frozen_initial_MaskAdapter'] and not s['model']['learned_768x768_value_transform'])
check('pool_phi_autograd_contract',s['inputs']['pool_then_phi'] and s['inputs']['phi_input_autograd_required'] and s['inputs']['phi_parameters_frozen'])
check('eta_and_energy_contract',s['model']['eta_max']==.5 and s['model']['eta_ramp_updates']==200 and s['model']['energy_bound']==4 and s['model']['divide_by_group_count'])
check('parameter_count_matches_reference',s['model']['expected_trainable_architecture_parameters']==216194)
roles=s['data']['roles'];fs={k:{v.split('_')[0] for v in vv} for k,vv in roles.items()}
check('fixed_data_roles',list(map(len,roles.values()))==[24,4,4] and not(fs['train']&fs['dev'] or fs['train']&fs['old_H'] or fs['dev']&fs['old_H']))
cf={v.split('_')[0] for v in s['cohorts']['scannet_cf18']};check('map_families_disjoint_from_learning',not(cf&(fs['train']|fs['dev']|fs['old_H'])))
check('cohort_counts',len(s['cohorts']['replica8'])==8 and len(s['cohorts']['scannet_cf18'])==18 and len(cf)==7)
check('supervision_scope',s['data']['base_classes']==160 and s['data']['adapter_heldout_classes']==40 and s['data']['train_base_originals']==622 and s['data']['dev_base_originals']==69)
check('H2_real_lock',not s['data']['h2']['annotations_before_predictor_lock'] and not s['data']['h2']['replace_names'] and not s['data']['real_proposals']['use_for_training_or_selection'])
check('training_count',s['training']['updates_per_arm']*4==s['training']['main_updates']==8000 and s['training']['main_updates']+s['training']['repeat_updates']==s['training']['max_scientific_updates']==16000)
check('training_batch_draw_scope',s['training']['effective_batch']==s['training']['microbatch']*s['training']['accumulation']==16 and s['training']['same_draws_all_arms'])
check('no_extra_losses',s['loss']['preservation']==s['loss']['correspondence']==0 and s['loss']['membership_BCE']==.10)
check('checkpoint_selection',s['training']['checkpoints']==[250,500,1000,1500,2000] and not s['training']['step0_nominatable'] and s['selection']['A_gain_min_fraction']==.005)
check('repeat_same_nominee',s['selection']['repeat_arms']=='ALL_FOUR' and s['selection']['transfer_trigger']=='PRENOMINEE_QUALIFIES_SEED29')
check('D1_cannot_select',s['diagnostics']['D1']['uses_GT'] and not s['diagnostics']['D1']['selects_methods'])
check('two_matched_controls',['MATCHED_MASS','MATCHED_MASS_AND_DELTA_NORM']==s['diagnostics']['interventions'][:2] and not s['diagnostics']['selection_allowed'])
m=s['whole_map'];check('map_coverage',len(m['main_methods'])==7 and len(m['repeat_methods'])==4 and len(set(m['main_methods']+m['repeat_methods']))==11 and len(m['main_methods'])*26==m['main_scene_rows']==182 and m['total_scene_rows']==286 and m['total_pools']==22)
check('whole_map_frozen_contract',m['replace_historical_F_only'] and m['recovered_labels_fixed'] and m['official_ranks_recomputed'] and not m['new_temperature'])
check('negative_statuses_valid',s['statuses']['no_main_signal']['exit']==0 and s['statuses']['no_main_signal']['new_map_rows']==0 and s['statuses']['not_repeated']['exit']==0 and s['statuses']['partial']['exit']==1)
check('runtime_not_gate',not s['latency_gate'] and s['resources']['cold_timing_grid']==0 and s['accuracy_priority'])
check('normal_push_actual_exit',s['publication']['normal_push_only'] and s['publication']['verify_full_local_remote_SHA'] and s['publication']['real_CLI_exit_observed'] and s['publication']['do_not_embed_own_future_commit_SHA'])
check('finite_validation',s['finite_validation']['max_groups']==12 and not s['finite_validation']['full_repo_suite'] and not s['finite_validation']['thousands_of_requirement_checks'])
rows=list(csv.DictReader((P/'EXPERIMENT_MATRIX.csv').open()));check('matrix_matches_spec',{r['id'] for r in rows if r['kind']=='trained_method'}=={m['id'] for m in methods})
check('syntax_parses',all(ast.parse((P/f).read_text()) is not None for f in ('reference_math.py','tests/test_reference_math.py')))
log=(P/'audit/REFERENCE_TEST_LOG.txt').read_text();check('reference_tests_passed','Ran 12 tests' in log and log.rstrip().endswith('OK'))
text='\n'.join((P/f).read_text() for f in required if f.endswith('.md'))
check('no_unresolved_design_placeholders',not re.search(r'\b(?:TODO|TBD|FIXME)\b',text))
check('essential_semantic_warnings',all(x in text for x in ('not correction vector norm','No R/MA checkpoint','not a strict global upper bound','No geometry-improvement claim')))
report={'status':'PASSED','scope':'EXECUTION_PACK_STATIC_AND_SYNTHETIC_MATH_ONLY','date':'2026-10-11',
        'checks':checks,'check_count':len(checks),'reference_unittest_count':12,
        'production_GPU_training_run':False,'production_FC_or_SAM2_loaded':False,'server_data_revalidated':False,
        'remote_repository_modified':False,'base_commit':s['base_commit'],
        'reference_log_sha256':hashlib.sha256(log.encode()).hexdigest()}
(P/'audit/AUDIT_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS',len(checks),'static checks; 12 reference tests; no production runs')
