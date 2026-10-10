"""Finite structural consistency audit for the instruction artifact."""
from pathlib import Path
import csv, hashlib, json, re, sys

ROOT = Path(__file__).resolve().parents[1]
s = json.loads((ROOT/'PROTOCOL_SPEC.json').read_text())
docs = {p.name:p.read_text() for p in ROOT.glob('*.md')}
checks=[]
def check(name, ok, detail=''):
    checks.append({'name':name, 'passed':bool(ok), 'detail':detail})

check('pinned_repository_commit', s['repository']=='Orangekostar/oviovo' and s['base_commit']=='6767eb90c2fd6999621b356b87269c012821713a')
check('isolated_new_branch', s['branch']=='research/ovimap-preservation-routing-v1')
check('accuracy_not_latency', not s['selection']['latency_gate'] and s['resources']['no_latency_objective'])
check('preserve_original_24_4_split', s['data']['reuse_exact_parent_split'] and s['data']['train_families']==24 and s['data']['dev_families']==4)
check('class_scopes', s['data']['base_classes']+s['data']['heldout_classes']==200)
check('old_H_explicitly_exposed', 'EXPOSED' in s['data']['old_H_role'])
check('H2_no_selection', not s['selection']['H2_or_regression_drives_selection'])
check('H2_four_reserved_families', s['data']['H2_families']==4 and 'nominations' in s['data']['H2_parse_annotations_after'])
check('four_R_factorial', {(r['residual'],r['preservation']) for r in s['R_methods']}=={(False,False),(True,False),(False,True),(True,True)})
check('no_old_failed_warmup', not s['R_model']['warmup_from_old_LR_weights'])
check('function_reference_required', 'initial_frozen' in s['R_model']['functional_residual'])
g={r['id']:r for r in s['G_methods']}
check('five_G_arms', len(g)==5)
check('G2_vs_G1_grouping_only', all(g['PR_G1_VIEW'][k]==g['PR_G2_SURFACE'][k] for k in ('membership_loss','direct_routing','correspondence_loss')))
check('G4_vs_G3_routing_only', all(g['PR_G3_AUX_ONLY'][k]==g['PR_G4_ROUTED'][k] for k in ('grouping','membership_loss','correspondence_loss')) and g['PR_G4_ROUTED']['direct_routing'])
check('G5_vs_G4_correspondence_only', all(g['PR_G5_CORRESP'][k]==g['PR_G4_ROUTED'][k] for k in ('grouping','membership_loss','direct_routing')) and g['PR_G5_CORRESP']['correspondence_loss'])
check('group_count_and_null', s['G_model']['group_count_normalization'] and s['G_model']['null_energy']==0)
check('no_renormalization_of_delta', not s['G_model']['normalize_local_delta_before_or_after_W0'] and 'bias=False' in s['G_model']['W0'])
check('training_total', 4*2000+2000+2*5*2000==s['optimizer']['maximum_updates']==30000)
check('first_step_not_nominee', s['optimizer']['step0_diagnostic_not_nomination'])
check('map_coverage', len(s['coverage']['main_map_methods'])==9 and len(s['coverage']['repeat_map_methods'])==6 and 26*15==s['coverage']['maximum_scene_rows'] and 2*15==s['coverage']['maximum_ordered_pools'])
check('real_proposal_cap', s['data']['real_proposal']['families']*s['data']['real_proposal']['frames_per_family']==8 and 8*s['data']['real_proposal']['retained_proposals_per_frame']==128)
check('FC_budget', s['resources']['maximum_FC_new_images_including_missing_parent_recovery']==3904+384+384)
check('new_backbone_calls_bounded', s['observation']['new_SAMV']==0 and s['observation']['new_AnyUp']==0 and s['observation']['new_ovi_maps']==0)
check('full_map_gate_unchanged', s['selection']['whole_map']['all5_vs_G1_nondecrease'] and s['selection']['whole_map']['CF_APall_gt_D2'] and s['selection']['whole_map']['tolerance_fraction']==1e-10)
check('normal_G_repeat_pairs', s['selection']['G_repeat'].startswith('all five'))
check('early_stop_documented', 'COMPLETE_NO_2D_FOUNDATION' in docs['CODEX_FINAL_EXECUTION_EN.md'] and 'COMPLETE_2D_NOT_REPEATED' in docs['EVALUATION_AND_RELEASE.md'])
check('real_readouts_not_oracle', 'Generate without GT points/boxes' in docs['DATA_AND_EXECUTION.md'] and 'Only after proposal inputs and head outputs lock' in docs['DATA_AND_EXECUTION.md'])
check('publication_actual_weights_curves', s['publication']['include_training_curves'] and not s['publication']['force_push'] and s['publication']['receipt'].startswith('external'))
check('four_reports', len(s['publication']['reports'])==4)
check('reference_test_pass', 'Ran 12 tests' in (ROOT/'audit/REFERENCE_TEST_LOG.txt').read_text() and '\nOK\n' in (ROOT/'audit/REFERENCE_TEST_LOG.txt').read_text())
links=[]
for name,text in docs.items():
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
        if not re.match(r'^[a-z]+://',link) and not link.startswith('#'):
            links.append((name,link,(ROOT/link.split('#')[0]).exists()))
check('local_document_links_exist', all(x[2] for x in links), repr(links))
with (ROOT/'EXPERIMENT_MATRIX.csv').open() as f:rows=list(csv.DictReader(f))
check('matrix_matches_spec', {r['method'] for r in rows}=={r['id'] for r in s['R_methods']+s['G_methods']})
check('bounded_test_groups', s['validation']['unit_groups_max']==12 and not s['validation']['new_full_repository_test_run'])
check('no_production_claim_from_checks', not s['validation']['scientific_metric_claim_from_reference_tests'])
report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL', 'checks':checks,
        'check_count':len(checks),'passed':sum(c['passed'] for c in checks),
        'reference_tests':12,'scope':'Instruction consistency and CPU toy numerical/autograd only',
        'production_training_run':False,'production_FC_or_SAM2_loaded':False,'repository_modified':False,
        'date':'2026-10-10'}
(ROOT/'audit/AUDIT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report['status'],report['passed'],'/',len(checks))
for c in checks:
    if not c['passed']: print('FAILED:',c)
sys.exit(0 if report['status']=='PASS' else 1)
