"""Reproducible specification/file consistency audit, not server validation."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parent
s=json.loads((R/'PROTOCOL_SPEC.json').read_text())
main=(R/'CODEX_FINAL_EXECUTION_EN.md').read_text()
contract=(R/'IMPLEMENTATION_CONTRACTS.md').read_text()
tab=(R/'TABLE_CONTRACTS.md').read_text()
checks=[]
def test(name,ok):checks.append({'check':name,'passed':bool(ok)})
required=['README.md','CODEX_FINAL_EXECUTION_EN.md','IMPLEMENTATION_CONTRACTS.md','TABLE_CONTRACTS.md',
          'CODE_REVIEW_AND_PLAN_ZH.md','SOURCE_EVIDENCE.md','PROTOCOL_SPEC.json','TABLE_BINDINGS.json','literature_reference.json',
          'replica8.txt','scannet_cf18.txt','reference_checks/check_contracts.py',
          'tables/table1_main.tex','tables/table2_ablation.tex','tables/table3_recovery.tex',
          'tables/table_layout_preview.tex','tables/table_layout_preview.pdf']
test('All required package files exist',all((R/p).is_file() for p in required))
test('Exact base SHA',bool(re.fullmatch('[0-9a-f]{40}',s['base_commit'])) and s['base_commit'] in main)
test('Exact upstream SHA',bool(re.fullmatch('[0-9a-f]{40}',s['upstream_commit'])) and s['upstream_commit'] in main)
test('Named release branch present',s['branch'] in main)
test('Replica list/order',len(s['cohorts']['replica8'])==8 and (R/'replica8.txt').read_text().splitlines()==s['cohorts']['replica8'])
test('CF18 list/order',len(s['cohorts']['scannet_cf18'])==18 and (R/'scannet_cf18.txt').read_text().splitlines()==s['cohorts']['scannet_cf18'])
test('Seven ScanNet physical families',len({x.split('_')[0] for x in s['cohorts']['scannet_cf18']})==7)
test('No dev scene silently replaces a benchmark scene',not set(s['development_scenes'])&set(s['cohorts']['scannet_cf18']))
test('Eight unique internal methods',len(s['methods'])==len({x['id'] for x in s['methods']})==8)
test('All IDs bound in main prompt',all(x['id'] in main for x in s['methods']))
test('172 fixed scientific scene-method records',sum(sum(len(s['cohorts'][c]) for c in m['cohorts']) for m in s['methods'])==172==s['scope']['main_scene_method_records'])
test('14 internal pools',sum(len(m['cohorts']) for m in s['methods'])==14==s['scope']['main_pool_records'])
test('26 common geometry maps not172',sum(map(len,s['cohorts'].values()))==s['scope']['main_anchor_maps']==26)
test('Only U2/G3 limited to Replica',all(m['cohorts']==(['replica8'] if m['short'] in ['U2','G3'] else ['replica8','scannet_cf18']) for m in s['methods']))
test('No new fit/model search',s['scope']['new_parameter_fits']==0 and not s['scope']['new_visual_backbone_search'])
b=json.loads((R/'TABLE_BINDINGS.json').read_text())
test('Shared table row bindings',b['table1']['measured_rows']==['CT_A0_NATIVE','CT_A1_E','CT_A3_ER'] and b['table2']['rows']==[m['id'] for m in s['methods'][:6]] and b['table3']['rows']==['CT_A1_E','CT_H_U2','CT_A3_ER','CT_G3'])
test('G1 independent of old requests',s['recovery']['new_view_selection_uses_historical_request_list'] is False and 'MUST NOT depend' in main)
test('G1/G3 nested fixed views',s['recovery']['g1_max_views']==1 and s['recovery']['g3_max_views']==3 and s['recovery']['g3_contains_g1_prefix'])
test('Explicit full-scene projection/ray-depth definition','not normalized' in s['projection']['ray'] and 'FULL predicted mesh' in main)
test('A5 unknown no disguised native fallback',s['recovery']['incumbent_fc_missing_label']==0 and 'No hidden Native fallback' in main)
test('Correct legacy metric overlap vector',len(s['evaluation']['ap_thresholds'])==9 and s['evaluation']['ap_thresholds'][-1]==.9)
test('Source and scoring min100 kept separate',s['recovery']['minimum_residual_source_rows']==100 and s['evaluation']['target_min_points']==100 and 'different spaces' in main)
test('One official primary rank mode',s['evaluation']['rank_mode']=='OFFICIAL_CURRENT_CLASS')
test('Three figures not additional task',s['scope']['main_tables']==3 and 'new figure-generation task' in tab)
test('24 single cold feature replays',len(s['timing']['arms'])*len(s['cohorts'][s['timing']['cohort']])*s['timing']['repeats_per_scene_arm']==24)
test('External reported context cannot become re-run claim',not s['external_rows']['exact_prediction_protocol_verified'] and not s['external_rows']['allow_new_external_model_install'])
test('Frozen A3, no benchmark winner renaming',s['selection']['primary_method']=='CT_A3_ER' and not s['selection']['change_primary_after_main_results'])
test('No automatic deployment',s['selection']['deployment']=='N0_UNCHANGED')
test('Normal push and external receipt',s['publication']['normal_push_only'] and not s['publication']['force_push'] and 'ls-remote' in main)
test('Reference checks actually executed', (R/'REFERENCE_CHECK_RESULTS.json').is_file() and json.loads((R/'REFERENCE_CHECK_RESULTS.json').read_text()).get('passed') is True)
# These are compilation/layout facts, not made-up experimental metrics.
pdf=R/'tables/table_layout_preview.pdf'
test('PDF layout preview produced',pdf.is_file() and pdf.read_bytes().startswith(b'%PDF-'))
for n in (1,2,3):
 p=R/'tables'/f'table{n}_{"main" if n==1 else "ablation" if n==2 else "recovery"}.tex'
 t=p.read_text()
 test(f'Table{n} keeps readable type and placeholders',r'\fontsize{9}' in t and '--' in t)
result={'scope':'EXECUTION_PACKAGE_SPECIFICATION_ONLY','checks':checks,'checks_count':len(checks),
        'passed_count':sum(x['passed'] for x in checks),'all_passed':all(x['passed'] for x in checks),
        'server_or_benchmark_run':False,'new_model_inference':0,'production_Cpp_compilation':False,
        'limitations':['No server data/model availability was tested','No G1 performance measured',
                       'External literature rows are not independently reproduced'],
        'files':{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in R.rglob('*') if p.is_file() and p.name not in ['PACKAGE_AUDIT.json'] and '__pycache__' not in str(p)}}
(R/'PACKAGE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['checks_count','passed_count','all_passed']},indent=2))
for c in checks:
 if not c['passed']:print('FAILED:',c['check'])
raise SystemExit(0 if result['all_passed'] else 1)
