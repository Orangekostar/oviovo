"""Audit the INSTRUCTION PACKAGE only; this does not test the proposed algorithms."""
from pathlib import Path
import json, hashlib, re

ROOT=Path(__file__).resolve().parent
PARTS=['00_CODEX_MASTER_EN.md','01_ASSETS_NATIVE_INTERFACES_EN.md','02_SEMANTIC_MODULE_EN.md',
       '03_GEOMETRY_MODULE_EN.md','04_QUERY_MODULE_EN.md','05_SELECTION_EVALUATION_RELEASE_EN.md']
HEADER='# FINAL EXECUTION INSTRUCTION — OVI-MAP module validation\n\nThis file concatenates the six authoritative English specification sections without changing their content. Companion PROTOCOL_SPEC.json and SOURCES.md are included in the same bundle.\n\n'
SEP='\n\n---\n\n'
checks=[]
def test(name,value):
    checks.append({'check':name,'pass':bool(value)})
    if not value:
        raise AssertionError(name)

required=PARTS+['CODEX_FINAL_EXECUTION_EN.md','PROTOCOL_SPEC.json','SOURCES.md','README_ZH.md','CODE_REVIEW_AND_AUDIT_ZH.md']
test('all_required_files_nonempty',all((ROOT/f).is_file() and (ROOT/f).stat().st_size>0 for f in required))
texts={f:(ROOT/f).read_text(encoding='utf-8') for f in required if f.endswith('.md')}
spec=json.loads((ROOT/'PROTOCOL_SPEC.json').read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
joined=HEADER+SEP.join(texts[f].rstrip() for f in PARTS)+'\n'
test('single_file_exactly_matches_modular_contract',(ROOT/'CODEX_FINAL_EXECUTION_EN.md').read_text()==joined)
test('json_strict_roundtrip',json.loads(json.dumps(spec,allow_nan=False))==spec)
test('reviewed_project_sha',spec['project_commit']=='b6455520c758a3413988e0827b3e1f34667bdfd1')
test('reviewed_upstream_sha',spec['official_commit']=='f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424')
test('wow_code_sha',spec['models']['wow']['code_commit']=='bfc6f2424c47097e70a641c6a8016319cac192cb')
d=spec['data'];test('independent_split_arithmetic',d['fit']+d['cal']+d['select']==d['development_families']==12 and d['confirm']==2)
test('new_capture_cap',d['development_families']+d['confirm']==d['max_new_independent_captures']==14)
test('three_head_dimensions',all(spec[k]['head_dims'][0]==spec[k]['feature_values']+spec[k]['feature_availability_bits'] for k in ['S','G','Q']))
test('semantic_methods_counts',len(spec['S']['direct_methods'])==5 and len(spec['S']['head_methods'])==3)
test('geometry_methods_counts',len(spec['G']['methods'])==3)
test('query_methods_counts',len(spec['Q']['methods'])==4)
ids=spec['S']['direct_methods']+spec['S']['head_methods']+spec['G']['methods']+spec['Q']['methods']
test('all_method_ids_unique',len(set(ids))==len(ids))
test('all_methods_documented',all(x in joined for x in ids))
test('teachers_documented',all(x in joined for x in spec['S']['teacher_candidates_CAL']))
test('fixed_semantic_budget',spec['S']['max_targets_per_scene']==128 and spec['S']['max_views_per_target']==3)
test('semantic_crop_budget_arithmetic',128*3*6==2304 and 128*3*3==1152)
test('fixed_geometry_budget',spec['G']['max_groups_per_scene']==64 and spec['G']['max_hypotheses_per_group']==8)
test('query_budgets',spec['Q']['main_budget_attempts']==200 and spec['Q']['curve_if_eligible']==[100,400])
test('no_fake_query_label_warmstart',not spec['Q']['free_final_native_labels'] and 'do not give it the final native label for free' in texts[PARTS[4]])
test('no_query_future_access',not spec['Q']['future_or_unacquired_features'] and 'prefix-sentinel' in texts[PARTS[4]])
test('static_G_declared',not spec['G']['live_feedback'] and 'STATIC' in texts[PARTS[3]])
test('mask_conditioned_WOW_only',not spec['models']['wow']['per_example_input_fallback'] and 'BLOCKED_WOW_MASK_INTERFACE' in texts[PARTS[2]])
test('missing_data_not_fabricated',d['supervised_missing_data']=='BLOCKED_INDEPENDENT_SCENES' and 'do not repurpose exposed scenes' in texts[PARTS[1]])
test('no_confirmation_reselection',not spec['selection']['post_confirm_retune'] and 'without changing thresholds' in texts[PARTS[5]])
test('release_code_results_handoff',all(spec['publication'][k] in joined for k in ['results_md','handoff_md','small_artifacts','patches']))
test('actual_push_required',spec['publication']['commit_and_push'] and spec['publication']['verify_remote_sha'] and 'git ls-remote' in texts[PARTS[5]])
test('no_force_or_auto_merge',not spec['publication']['force_push'] and not spec['publication']['auto_merge_main'])
test('json_disabled_thresholds_explicit',spec['S']['adoption_thresholds_CAL'][-1]=='KEEP_ALL' and spec['G']['adoption_margins_CAL'][-1]=='KEEP_ALL')
test('no_old24plus4_capture_contract','take at most 24' not in joined and 'At most 28 new captures' not in joined)
test('no_unfilled_placeholder_tokens',not re.search(r'\b(TODO|TBD|PLACEHOLDER|INSERT_HERE)\b',joined))
test('bounded_training',spec['training']['max_epochs']==100 and not spec['training']['architecture_or_seed_grid'])
test('spec_not_claimed_runtime_config',spec['specification_only'] is True)
result={'audit_scope':'instruction package consistency, NOT algorithm execution',
        'version':spec['version'],'checks_passed':len(checks),'checks':checks,
        'algorithm_experiments_executed_here':False,'new_model_weights_loaded_here':False,'github_push_performed_here':False,
        'authoritative_file_sha256':hashlib.sha256(joined.encode()).hexdigest(),
        'external_dependencies_still_runtime_bound':['authorized independent scenes','native loaded extension export','WOW weights and actual mask interface']}
(ROOT/'PACKAGE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['audit_scope','checks_passed','algorithm_experiments_executed_here','authoritative_file_sha256']},indent=2))
