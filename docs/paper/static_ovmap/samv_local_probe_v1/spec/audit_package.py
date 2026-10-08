"""Bounded specification audit for this instruction package, not production checks."""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
spec = json.loads((ROOT/'PROTOCOL_SPEC.json').read_text())
checks=[]
def check(name, condition):
    checks.append({'name':name,'passed':bool(condition)})
    if not condition:
        raise AssertionError(name)

scenes=[s for names in spec['cohorts'].values() for s in names]
check('exact_four_scenes',scenes==['office1','room0','scene0011_00','scene0050_00'])
check('unique_scenes',len(set(scenes))==4)
check('six_main_arms',len(spec['methods'])==6)
check('unique_method_ids',len({x['id'] for x in spec['methods']})==6)
check('main_coverage',len(scenes)*len(spec['methods'])==spec['scope']['main_scene_rows']==24)
check('total_coverage',24+4==spec['scope']['all_scene_rows']==28)
check('subset_pools',6*2+2==spec['scope']['all_pools']==14)
check('targets_bound',4*spec['scope']['targets_per_scene_max']==spec['scope']['targets_total_max']==32)
check('observation_bound',4*32==spec['scope']['observer_frames_total_max']==128)
check('joint_frame_slots',32*spec['views']['count_primary']==spec['scope']['samv_scientific_frame_slots_max']==192)
check('fc_request_bound',32*2*2==spec['scope']['semantic_region_pools_max']==128)
check('fc_image_bound',32*2==spec['scope']['new_FC_images_max']==64)
check('timing_bound',2*len(spec['cost']['methods'])*spec['cost']['repeats']==spec['scope']['extra_segmentation_timing_calls']==8)
check('no_unapproved_expansion',spec['scope']['full_26_scene_expansion'] is False and spec['selection']['automatic_expansion'] is False)
check('no_new_training_mapping',spec['scope']['new_maps']==spec['scope']['new_training']==spec['scope']['new_AnyUp']==spec['scope']['new_NQ']==0)
check('pinned_commits',all(re.fullmatch('[0-9a-f]{40}',v) for v in [spec['base_commit'],spec['assets']['samv']['commit'],spec['assets']['sam2']['commit']]))
check('unknown_asset_hash_honest',spec['assets']['samv']['weights_sha256'] is None and spec['resources']['samv_expected_asset_bytes_not_claimed'])
check('isolated_incompatible_envs',spec['assets']['samv']['torch']=='2.3.1' and spec['assets']['sam2']['torch']=='2.5.1' and spec['assets']['sam2']['separate_environment'])
check('sam2_not_default_bplus',spec['assets']['sam2']['config'].endswith('_l.yaml') and 'large' in spec['assets']['sam2']['weights'])
check('explicit_sam2_postprocessing',not spec['assets']['sam2']['builder_apply_postprocessing'] and spec['assets']['sam2']['hydra_overrides_extra'][-1]=='++model.fill_hole_area=0')
check('no_gt_model_prompts',not spec['prompts']['box_or_GT_or_text_prompts'] and not spec['evaluation']['GT_prediction_inputs'])
check('global_resource_profile',spec['views']['count_resource_fallback']==4 and 'GLOBAL' in spec['views']['fallback_policy'])
check('finite_main_pred_contract',spec['repair']['min_visible_views']==2 and spec['repair']['positive_rate']==2/3 and spec['repair']['negative_rate']==1/3)
check('single_partition',spec['evaluation']['one_exclusive_partition'] and spec['evaluation']['rebuild_all_masks_for_actual_partition'])
check('paired_semantics',spec['semantics']['same_joint_success_domain'] and spec['semantics']['required_successful_views_per_mask_source']==2)
check('combined_not_new_inference',spec['methods'][-1]['partition']=='EXACT_SV02' and spec['methods'][-1]['labels']=='EXACT_SV04_CLASS_MAP')
check('valid_primary_candidates',set(spec['selection']['eligible_methods'])<=set(x['id'] for x in spec['methods']))
check('limited_tests',spec['resources']['new_production_tests_target_max']==12 and not spec['resources']['full_repo_tests'])
check('synchronized_cost_parity',spec['cost']['raw_mask_parity_required'] and not spec['cost']['feature_and_result_cache'])
check('scope_terms_in_main',all(x in (ROOT/'CODEX_FINAL_EXECUTION_EN.md').read_text() for x in ['PUSH_VERIFIED','partial handoff','NO_ANCHOR_SUPPORT','No automatic 26-scene expansion','SV03']))
check('new_support_permission_documented','raw owner=0 is NOT a universal' in (ROOT/'IMPLEMENTATION_CONTRACTS.md').read_text())
check('file_set_present',all((ROOT/f).is_file() for f in ['README.md','CODEX_FINAL_EXECUTION_EN.md','CODE_REVIEW_AND_PLAN_ZH.md','IMPLEMENTATION_CONTRACTS.md','EVALUATION_AND_SELECTION.md','EXPERIMENT_MATRIX.md','SOURCE_EVIDENCE.md','tables/TABLE_DESIGN.md']))
for f in (ROOT/'reference').glob('*.py'): ast.parse(f.read_text())
check('reference_python_parses',True)
proc=subprocess.run([sys.executable,'-m','unittest','-v','test_rules'],cwd=ROOT/'reference',capture_output=True,text=True)
log=proc.stdout+proc.stderr
(ROOT/'REFERENCE_TEST_LOG.txt').write_text(log)
check('12_synthetic_rules_pass',proc.returncode==0 and 'Ran 12 tests' in log and '\nOK\n' in log)
report={'artifact':'INSTRUCTION_PACKAGE_AUDIT','date':'2026-10-08','status':'PASS',
 'checks_passed':len(checks),'checks':checks,'synthetic_test_count':12,
 'synthetic_test_returncode':proc.returncode,
 'manual_review':['GT-derived upstream CLI excluded','Six-to-four only prefreeze real OOM',
  'Different Torch environments isolated','Single source-row partition for all metrics',
  'Original-mask paired FC control added','Protected-core and raw-mask suppression separately diagnosed',
  'Stage-2 author checkpoint not claimed downloaded','No full benchmark expansion',
  'Code/compact results normal push and exact remote SHA proof'],
 'not_executed':['SAM-V model loading/inference','SAM2 inference','server asset/data verification',
 'new scene AP/mIoU evaluation','new runtime/VRAM benchmark','GitHub writes'],
 'scope_warning':'PASS verifies this specification and synthetic rules only, not a scientific gain.'}
(ROOT/'AUDIT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks_passed':len(checks),'synthetic_tests':12},ensure_ascii=False))
