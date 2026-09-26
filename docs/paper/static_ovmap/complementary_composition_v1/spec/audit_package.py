#!/usr/bin/env python3
"""Validate this instruction package only. Does not run repository code/models."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main() -> None:
    checks: list[dict[str, str]] = []
    def check(name: str, condition: bool) -> None:
        if not condition:
            raise AssertionError(name)
        checks.append({'check': name, 'result': 'PASS'})

    required = ['CODEX_FINAL_EXECUTION_EN.md', 'PROTOCOL_SPEC.json',
                'CODE_REVIEW_AND_AUDIT_ZH.md', 'SOURCES.md', 'README.md', 'audit_package.py']
    check('required_files_exist_and_nonempty', all((ROOT / f).is_file() and (ROOT / f).stat().st_size > 0 for f in required))
    spec = json.loads((ROOT / 'PROTOCOL_SPEC.json').read_text())
    en = (ROOT / 'CODEX_FINAL_EXECUTION_EN.md').read_text()
    zh = (ROOT / 'CODE_REVIEW_AND_AUDIT_ZH.md').read_text()
    sources = (ROOT / 'SOURCES.md').read_text()
    check('reviewed_revision_is_full_sha', bool(re.fullmatch(r'[0-9a-f]{40}', spec['reviewed_commit'])))
    check('reviewed_revision_matches_instruction', spec['reviewed_commit'] in en and spec['reviewed_commit'] in zh)
    check('spec_is_not_pretended_runnable_config', spec['specification_only'] and 'resolved_config.json' in en)
    check('phase_names_in_instruction', all(p in en for p in spec['phases']))
    check('fixed_mandatory_matrix', spec['mandatory'] == ['M1','M2_RAW','M2_CAL','M3','M4','M5'])
    check('unique_method_ids', len({m['id'] for m in spec['methods'].values()}) == len(spec['methods']))
    check('method_ids_exist_in_instruction', all(m['id'] in en for m in spec['methods'].values()))
    check('fixed_geometry_checks_remain_active', all(m['payload_branch'] in {'S','Q'} for m in spec['methods'].values()))
    groups = [spec['data'][k] for k in ('compose_cal','regression_only','confirmation')]
    check('three_disjoint_two_scene_roles', all(len(g)==2 for g in groups) and len(set(sum(groups, [])))==6)
    check('role_ids_explicit_in_instruction', all(x in en for g in groups for x in g))
    check('24_mandatory_new_rows', spec['mandatory_new_rows_before_confirmation'] == len(spec['mandatory']) * sum(len(g) for g in groups[:2]) == 24)
    check('4_conditional_rows', spec['conditional_new_rows_before_confirmation'] == sum(len(g) for g in groups[:2]) == 4)
    check('m6_not_old_standalone_gate', spec['gates']['no_individual_retention_prerequisite'] and 'weak_dom(M5, Q_COMBINE)' in en)
    check('single_nominee_max_six_holdout_methods', spec['gates']['confirm_candidate_count'] == 1 and spec['gates']['confirmation_max_methods'] == 6)
    check('frozen_native_controllers_no_s2_feedback', not spec['query']['siglip2_feedback_to_controller'] and 'paid-attempt' in en and 'must never change the controller' in en)
    check('B200_and_single_shared_state', spec['query']['budget']==200 and spec['query']['shared_state'])
    c=spec['calibration']
    check('temperature_default_inside_bounds', c['temperature_bounds'][0] < c['default_temperature'] < c['temperature_bounds'][1])
    check('only_finite_scalar_calibration', c['maxiter']==64 and c['min_objects_per_source_fit']==5 and c['min_classes_per_source_fit']==2)
    check('crossfit_then_freeze_no_holdout_fit', 'leave-one-COMPOSE_CAL-scene-out' in en and 'fit temperatures once using both CAL scenes' in en)
    r=spec['runtime']
    check('native_request_budget_arithmetic', r['cal_reg_max_trajectory_native_requests']==4*3*spec['query']['budget'])
    check('s2_request_budget_arithmetic', r['cal_reg_max_trajectory_siglip2_requests_including_m6']==4*3*spec['query']['budget'])
    check('static_request_budget_arithmetic', r['cal_reg_max_static_siglip2_requests']==4*r['static_siglip2_max_targets']*r['static_siglip2_max_views'])
    check('no_new_background_or_head_training', r['native_background_crops_new']==0 and r['no_new_head_training'])
    check('current_area_fix_reflected', 'already repaired' in en and '已修复' in zh and 'Current AREA preserves raw' in sources)
    check('obsolete_area_claim_removed', 'Existing static AREA normalizes view means' not in sources)
    check('raw_fallback_is_not_independent_vote', 'fallback is **not** a second independent vote' in en)
    check('explicit_trace_retention_and_model_separation', 'permanently drops' in en and 'separate stores per visual model' in en)
    check('truthful_method_and_physical_costs', 'Method-required logical work' in en and 'Physical work in this task' in en)
    check('four_required_tables', all(f'**Table {c} —' in en for c in 'ABCD'))
    check('both_reports_and_push_verification', all(x in en for x in ('COMPOSITION_RESULTS.md','COMPOSITION_HANDOFF.md','git ls-remote','PUSH_VERIFIED')))
    check('no_old_release_copy', not spec['release']['copy_previous_16430_files'] and 'Do **not** copy the 16,430 files' in en)
    check('scientific_status_not_test_count', 'no test-count target' in en.lower() and 'Never present a package audit or test pass count as a benchmark gain' in en)
    check('no_editorial_placeholders', not re.search(r'\b(?:TODO|TBD|FILL_THIS|INSERT_HERE)\b', en))
    check('all_source_ids_documented', all(f'SRC{i:02d}' in sources for i in range(1,24)))
    check('primary_calibration_and_model_sources', 'proceedings.mlr.press/v70/guo17a.html' in sources and 'huggingface.co/google/siglip2-large-patch16-384' in sources)
    manifest={name:{'bytes':(ROOT/name).stat().st_size, 'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in required}
    report={'artifact_type':'INSTRUCTION_PACKAGE_AUDIT','status':'PASS','scope':'Package structure and explicit contract consistency only; no model or benchmark was executed.',
            'reviewed_revision':spec['reviewed_commit'],'checks':checks,'files':manifest,
            'manual_review':{'scope_and_phase_dependencies':'REVIEWED','CAL_vs_regression_vs_confirmation':'REVIEWED',
             'current_AREA_normalization_corrected_after_source_reread':'REVIEWED_AND_CORRECTED',
             'fixed_trajectory_vs_online_model_change':'REVIEWED','fallback_and_lineage':'REVIEWED',
             'logical_cost_vs_cache_reuse':'REVIEWED','new_confirmation_contract_vs_old_gate':'REVIEWED',
             'release_identity_and_no_self_reference':'REVIEWED'},
            'runtime_verification':{'server_asset_readability':'NOT_PERFORMED_HERE','new_method_execution':'NOT_PERFORMED','remote_repository_mutation':'NONE'}}
    (ROOT/'PACKAGE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','package_checks':len(checks),'benchmark_executed':False},ensure_ascii=False))

if __name__=='__main__':
    main()
