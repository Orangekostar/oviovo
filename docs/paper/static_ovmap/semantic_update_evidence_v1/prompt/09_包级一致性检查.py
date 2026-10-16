"""Checks generated instructions only; it does not validate server experiments."""
from pathlib import Path
import json, ast, re, hashlib
P=Path(__file__).resolve().parent
j=json.loads((P/'02_冻结执行配置.json').read_text())
rows=[]
def check(name,value):
    if not value:raise AssertionError(name)
    rows.append({'check':name,'status':'PASS'})
check('base_and_namespace',j['base_commit']=='553758bda89031b05916ac81a2981b597dfb3bda' and j['namespace'].endswith('/semantic_update_evidence_v1'))
check('A_110_logical_candidates',len(j['A']['gamma_grid'])*(len(j['A']['alpha_grid'])+len(j['A']['angle_cap_degrees']))==110)
check('B_3_arms_2_seeds',len(j['B']['methods'])*len(j['B']['seeds'])*j['B']['train_updates_per_arm']==12000)
check('B_main_budget',12000*32+6*5*69*4+200*32<=j['runtime']['max_B_train_validation_head_calls'])
check('fresh_GT_REAL_budget',768*4*(14*3+6)+4608*14<=j['runtime']['max_C_head_calls'])
check('known_REG_input_budget',1436*14<=j['runtime']['max_REG_head_calls'])
check('class_partition',j['B']['core_class_count']+j['B']['pseudo_class_count']+j['B']['original_novel_class_count']==200)
check('caps_and_scoring',12*96==1152 and 12*12==144 and 14*12==168 and 14*2*26==728)
check('projection_is_not_mislabelled',j['B']['exact_intersection_projection'] is False)
check('conditional_scope',j['B']['enabled_only_after']=='A_GATE_OPEN' and j['C']['enabled_only_after']=='B_GATE_OPEN')
for name in ['00_给Codex的最终执行指令.md','01_源码证据与任务绑定.md','03_数学与选模精确定义.md','04_数据评分与发布接口.md']:
    text=(P/name).read_text();check(name+'_codefences',text.count('```')%2==0)
for name in ['06_算子参考与合同自检.py','09_包级一致性检查.py']:
    ast.parse((P/name).read_text());check(name+'_parse',True)
source=json.loads((P/'08_源码证据清单.json').read_text())
check('source_ids_unique',len({r['id'] for r in source})==len(source))
check('declared_shas_format',all(re.fullmatch('[0-9a-f]{40}',r['git_blob_sha']) for r in source))
main=(P/'00_给Codex的最终执行指令.md').read_text()
check('main_matches_updated_caps','REG≤24000，C≤220000' in main)
check('static_profile_not_invented','不虚构matched1000列' in main)
result={'status':'PASS','scope':'PACKAGE_STATIC_CONSISTENCY_ONLY','checks':rows,'check_count':len(rows),'real_execution_validated':False}
(P/'10_包级检查结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
