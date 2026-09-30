"""Audit this instruction package, not a repository experiment; no network calls."""
from pathlib import Path
import ast
import hashlib
import io
import json
import re
import sys
import unittest

P=Path(__file__).resolve().parent
spec=json.loads((P/'PROTOCOL_SPEC.json').read_text())
main=(P/'CODEX_FINAL_EXECUTION_EN.md').read_text()
contracts=(P/'IMPLEMENTATION_CONTRACTS.md').read_text()
review=(P/'CODE_REVIEW_AND_PLAN_ZH.md').read_text()
sources=(P/'SOURCE_EVIDENCE.md').read_text()
checks=[]
def check(name, ok, detail=''):
    checks.append({'name':name,'passed':bool(ok),'detail':detail})

required=['README.md','CODEX_FINAL_EXECUTION_EN.md','CODE_REVIEW_AND_PLAN_ZH.md',
          'IMPLEMENTATION_CONTRACTS.md','SOURCE_EVIDENCE.md','PROTOCOL_SPEC.json',
          'ACCEPTANCE_AND_AUDIT.md','reference_kernels.py','test_reference_kernels.py','audit_package.py']
check('all_required_package_files_present', all((P/x).is_file() for x in required))
check('fixed_user_and_upstream_commits',spec['base_commit']=='1ce806b22c63943843300006b5b1034ec9e1cb0b' and spec['upstream_commit']=='f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424')
ids=[v['id'] for v in spec['map_variants']]
check('seven_unique_primary_map_variants',len(ids)==len(set(ids))==7)
check('variant_families_1_control_4_structural_2_frontend',[sum(v['family']==x for v in spec['map_variants']) for x in ['control','structural','frontend']]==[1,4,2])
check('all_variant_ids_bound_in_instruction',all(x in main for x in ids))
d=spec['datasets'];dev=d['development'];rep=d['replica']
check('four_dev_eight_replica_unique_disjoint',len(set(dev))==4 and len(set(rep))==8 and not set(dev)&set(rep))
check('no_confirmation_scenes_in_executable_cohort',not set(d['old_confirm_scenes_not_authorized']) & set(dev+rep) and d['independent_confirmation']=='NOT_RUN_OUT_OF_SCOPE')
check('all_scenes_disclosed_as_exposed',d['all_exposed'] is True and 'historically exposed' in main)
check('original_cal_folds_remain_in_dev',set(d['parent_calibration']).issubset(dev) and len(d['parent_calibration'])==2)
check('native_mode4_and_original_core_modes_fixed',spec['native_modes']['inst_association']==4 and spec['native_modes']['data_association']==2 and spec['native_modes']['seg_graph_confidence']==3 and 'Keep `inst_association=4`' in main)
check('ratio_is_separate_control_not_global_repair',spec['association']['ratio_control'].startswith('ENABLE_EXISTING') and 'Do not enable this change in FORWARD/BIDIR' in contracts)
check('all_depth_thresholds_and_original_gate_position_fixed',spec['depth_fusion']['min_depth_segment_pixels']==100 and spec['depth_fusion']['split_candidate_coverage_strict_gt']==.9 and spec['depth_fusion']['split_depth_fraction_strict_lt']==.5 and spec['depth_fusion']['residual_label_fraction_ge']==.2)
check('sam2_pinned_current_frame_only_and_bounded',spec['frontend']['sam2_commit']=='2b90b9f5ceec907a1c18123530e92e794ad901a4' and spec['frontend']['chunk_valid_frames']==5 and spec['frontend']['max_frame_num_to_track']==0 and spec['frontend']['future_prompts'] is False)
check('no_silent_model_family_substitution',spec['frontend']['selected_family']=='B03_SAM2' and spec['frontend']['automatic_family_fallback'] is False and 'B04' in spec['scope']['deferred'])
check('real_new_inference_authorized_training_refit_disabled',spec['runtime']['new_visual_inference'] is True and spec['runtime']['fundamental_model_training'] is False and spec['runtime']['new_temperature_fit'] is False and spec['runtime']['new_Q_GAIN_training'] is False)
check('three_frozen_readouts_and_d2_weights',spec['semantics']['readouts']==['NATIVE_READOUT','FC_EQ','D2'] and spec['semantics']['D2_all_present_weights']=={'N':.25,'Q':.25,'F':.5})
check('native_and_fc_observation_rules_fixed',spec['semantics']['Q_budget_attempts_per_scene']==200 and spec['semantics']['native_saved_max_views']==10 and spec['semantics']['native_readout_last_views']==8 and spec['semantics']['FC_views_per_target']==3 and spec['semantics']['FC_target_cap']==128)
check('fresh_geometry_anchor_and_projection_required',spec['semantics']['fresh_native_anchor_per_map'] is True and spec['semantics']['copy_old_owner_scores'] is False and 'new** map' in main and 'Regenerate Open3D float32 1NN projection' in main)
check('official_pool_and_actual_overlap_vector_fixed',spec['evaluation']['official_pool']=='RELEASED_DATASET_POOL' and '0.50_TO_0.90' in spec['evaluation']['apall_overlap_vector'] and spec['evaluation']['projection_distance_strict_lt_m']==.05)
check('one_composition_no_individual_gain_gate',spec['selection']['max_combinations']==1 and spec['selection']['require_individual_net_gain_for_combination'] is False)
check('map_job_and_primary_row_upper_bounds', (len(ids)*len(dev)+len(dev)+spec['selection']['max_replica_map_variants']*len(rep))==spec['scope']['maximum_successful_map_scene_jobs']==64 and spec['scope']['maximum_official_primary_rows']==64*3)
check('selection_is_development_only_frozen_for_transfer',spec['selection']['refit_after_replica'] is False and spec['selection']['threshold_grid'] is None and spec['selection']['APall_band_pp']==.05 and spec['selection']['mIoU_band_pp']==.1)
check('shared_cache_cost_not_zero_standalone_cost','STANDALONE_REQUIRED' in spec['selection']['tie_order'][0] and 'standalone' in contracts.lower())
check('all_phases_and_cli_agree',all(x in main for x in spec['phases']) and all(x in main for x in ['--mapping-workers','--mapping-threads','--evaluation-workers','--spec','--output-root','--gpu']))
check('reports_and_real_normal_push_bound',len(spec['publication']['reports'])==4 and all(x in main for x in spec['publication']['reports']) and spec['publication']['force_push'] is False and 'git ls-remote --heads origin refs/heads/research/ovimap-backbone-wave1-v1' in main)
check('production_and_package_validation_distinguished','not native/GPU validation' in main and 'does not' in (P/'ACCEPTANCE_AND_AUDIT.md').read_text())
check('source_registry_has_pinned_real_integration_files',all(x in sources for x in ['S01','S19','U01','U10','semantic_instance_label_fusion.cc','sam2_video_predictor.py']))
check('eleven_inherited_2026_references_not_all_executed',sources.count('— CVPR2026')+sources.count('— ICLR2026')==11 and 'SAM2.1' in sources)
check('no_unresolved_editorial_placeholders',not re.search(r'\b(TODO|TBD|FIXME)\b',main+'\n'+contracts))
for fn in ['reference_kernels.py','test_reference_kernels.py','audit_package.py']:
    ast.parse((P/fn).read_text(),filename=fn)
check('python_fixture_and_audit_syntax',True)

sys.path.insert(0,str(P))
suite=unittest.defaultTestLoader.discover(str(P),pattern='test_reference_kernels.py')
log=io.StringIO()
result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
(P/'REFERENCE_TEST_LOG.txt').write_text(log.getvalue())
check('twelve_actual_synthetic_tests_passed',result.testsRun==12 and result.wasSuccessful())
report={
 'task_id':spec['task_id'],'issued':'2026-09-30',
 'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
 'scope':'INSTRUCTION_PACKAGE_AND_SYNTHETIC_MATHEMATICS_ONLY',
 'not_performed':['native_C++_compilation','SAM_or_VLM_inference','server_cache_access','scene_reconstruction','benchmark_evaluation','remote_repository_write'],
 'source_review':'Pinned GitHub source/hand-off/parameter reads and roadmap comparison; see SOURCE_EVIDENCE.md.',
 'checks_total':len(checks),'checks_passed':sum(c['passed'] for c in checks),
 'checks':checks,
 'reference_tests':{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'log':'REFERENCE_TEST_LOG.txt'}
}
(P/'PACKAGE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':report['checks_total'],'passed':report['checks_passed'],'synthetic_tests':result.testsRun},ensure_ascii=False))
if report['status']!='PASS':
    print('\n'.join(c['name'] for c in checks if not c['passed']))
    raise SystemExit(1)
