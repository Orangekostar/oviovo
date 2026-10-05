"""Audit this execution package only. No production/GPU/data evaluation is implied."""
from pathlib import Path
import ast, json, re, subprocess, sys

ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'PROTOCOL_SPEC.json').read_text())
e=json.loads((ROOT/'SOURCE_EVIDENCE.json').read_text())
main=(ROOT/'CODEX_FINAL_EXECUTION_EN.md').read_text()
contracts=(ROOT/'IMPLEMENTATION_CONTRACTS.md').read_text()
timing=(ROOT/'TIMING_AND_TABLES.md').read_text()
checks=[]
def check(name, value, scope='SPECIFICATION'):
    checks.append({'check':name,'passed':bool(value),'scope':scope})
required=['README.md','CODEX_FINAL_EXECUTION_EN.md','IMPLEMENTATION_CONTRACTS.md',
          'TIMING_AND_TABLES.md','PROTOCOL_SPEC.json','CODE_REVIEW_AND_PLAN_ZH.md',
          'SOURCE_EVIDENCE.md','SOURCE_EVIDENCE.json','TABLE_TEMPLATES.tex',
          'reference/invariants.py','reference/test_invariants.py','tables/LAYOUT_PREVIEW.pdf']
check('required_files',all((ROOT/f).is_file() for f in required))
check('actual_v2_required',s['source_policy']['require_actual_measured_v2'])
check('missing_v2_not_reimplemented',not s['source_policy']['allow_reimplementing_missing_fallback'])
check('remote_v2_not_claimed',not s['source_policy']['remote_contains_v2_at_authoring'])
check('same_inspected_commit',s['inspected_commit']==e['inspected_commit'])
check('replica8_unique',len(s['cohorts']['replica8'])==len(set(s['cohorts']['replica8']))==8)
check('cf18_unique',len(s['cohorts']['scannet_cf18'])==len(set(s['cohorts']['scannet_cf18']))==18)
check('cf18_seven_families',len({p.split('_')[0] for p in s['cohorts']['scannet_cf18']})==7)
check('pilots_are_predeclared_members',s['pilot_scenes']==['room0','office1'] and set(s['pilot_scenes'])<=set(s['cohorts']['replica8']))
check('four_execution_variants',len(s['variants'])==4 and len({r['id'] for r in s['variants']})==4)
check('pilot_call_budget',len(s['pilot_scenes'])*len(s['variants'])*s['pilot']['repeats']==s['pilot']['calls_max']==16)
check('candidate_final_call_budget',len(s['cohorts']['replica8'])*len(s['timing']['arms_if_candidate'])*s['timing']['repeats']==s['timing']['calls_if_candidate']==64)
check('reference_final_call_budget',len(s['cohorts']['replica8'])*len(s['timing']['arms_if_reference_retained'])*s['timing']['repeats']==s['timing']['calls_if_reference_retained']==48)
check('normal_total_budget',s['pilot']['calls_max']+s['timing']['calls_if_candidate']==s['limits']['max_cold_recovery_calls_normal']==80)
check('zero_maps_fits_models',all(s['science'][k]==0 for k in ['new_maps','new_fits','new_backbones','new_benchmarks','new_native_crop_inference','new_segmentation_inference','new_text_encoding']))
check('no_new_neural_batching',not s['limits']['new_regional_or_image_model_batching'] and s['limits']['gpu_encoder_batch_size']==1)
check('parent_scientific_grid',s['science']['parent_scene_outputs']==172 and s['science']['parent_pools']==14)
check('three_main_tables',s['reporting']['main_tables_remain']==3)
check('shared_readout_rows',s['science']['readout_reuse']==['CT_A2_R','CT_A5_FC_ONLY','CT_A3_ER'])
check('g1_g3_prefix',s['science']['g1_views']==1 and s['science']['g3_views']==3 and s['science']['g1_is_g3_prefix'])
check('prediction_parity_required',s['verification']['labels_masks_ranks_availability_exact'] and s['verification']['compare_by_scientific_content_key'])
check('cf18_not_falsely_live_timed',s['verification']['cf18_cached_export_parity_only'] and 'not CF18 live optimized-model timing' in main)
check('new_publication_branch',s['branch']=='research/ovimap-runtime-parity-v1' and s['branch'] in main)
check('immutable_reference_controls','Do not edit the old freeze' in main and 'snapshot' in main)
check('cold_controls_remeasured','U2_CONTROL' in main and 'old 45.58/45.63/6.61 values are context' in main)
check('no_guaranteed_speedup','not a statistical test' in main and 'Faster execution never establishes a new accuracy gain' in main)
check('new_request_ids_not_forged','Do not forge the old source hash' in ' '.join(contracts.split()) and 'Execution identity' in contracts)
check('sparse_diagnostic_scope','queried_pixels' in contracts and 'Global hits outside queried pixels are null' in contracts)
check('full_scene_occlusion','unchanged complete scene' in main and 'noncandidate object must be able to occlude' in contracts)
check('near_plane_fallback','camera is inside it' in contracts and 'use the entire frame' in contracts)
check('parent_model_not_320_resize','800/1333 resize' in contracts)
check('no_old_result_timing_reads','no access to parent recovery features/results' in contracts)
check('latency_never_minimum','never the minimum' in main and 'average ratios instead of aggregate latency ratio' in timing)
check('precise_scope_of_template','Templates only' in (ROOT/'TABLE_TEMPLATES.tex').read_text())
check('code_ast_parse',all(ast.parse(p.read_text()) is not None for p in (ROOT/'reference').glob('*.py')), 'REFERENCE_CODE_SYNTAX')
proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','reference','-p','test_*.py','-v'],cwd=ROOT,text=True,capture_output=True)
log=proc.stdout+proc.stderr
(ROOT/'REFERENCE_TEST_LOG.txt').write_text(log)
count=re.search(r'Ran (\d+) tests?',log)
check('reference_examples_pass',proc.returncode==0,'REFERENCE_ONLY_NOT_PRODUCTION')
import fitz
pdf=fitz.open(ROOT/'tables/LAYOUT_PREVIEW.pdf')
check('table_preview_one_page',len(pdf)==1,'PDF_LAYOUT')
check('table_preview_is_labeled_placeholder','templates, not measured results' in pdf[0].get_text().lower(),'PDF_LAYOUT')
report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL',
 'scope':'Execution specification consistency, reference examples and preview layout; not real v2 source, Open3D, CUDA, scene metrics or runtime validation.',
 'check_count':len(checks),'checks':checks,'reference_test_count':int(count.group(1)) if count else None,
 'production_models_run':0,'server_experiments_run':0,'github_writes':0,
 'actual_v2_source_reviewed':False,
 'visual_review':'Preview rendered with pdftoppm and visually inspected; title order corrected; no clipping seen.',
 'manual_review_corrections':[
  'Bind real v2 rather than silently use incomplete remote v1.',
  'Compare scientific request content, not producer-bound opaque IDs.',
  'Sparse full-scene query diagnostics scoped; no fabricated global totals.',
  'Readout controls reused, not mislabeled as backbone-only ablation.',
  'New source-aware timing ledger allows explicit repeats without mutating old reserved calls.',
  'No neural arithmetic batching or lower-precision speed shortcuts.',
  'Same new timing boundary and contemporary U2 control; historical times are context.',
  'CF18 cache/export verification is not claimed as new live GPU timing.',
  'Ground-truth labels excluded from prediction; inherited evaluation projection remains an export adapter.',
  'Normal timing count 16+64=80, or reference-retained reduced scope; no whole-map rerun.'
 ]}
(ROOT/'AUDIT_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'reference_tests':report['reference_test_count'],'failed':[c['check'] for c in checks if not c['passed']]},indent=2))
sys.exit(0 if report['status']=='PASS' else 1)
