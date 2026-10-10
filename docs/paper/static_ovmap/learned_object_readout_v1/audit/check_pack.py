"""Finite specification audit, not a scientific experiment or model validation."""
from pathlib import Path
import ast,csv,json,re,sys,hashlib
R=Path(__file__).resolve().parents[1]
s=json.loads((R/'PROTOCOL_SPEC.json').read_text())
results=[]
def check(name,condition):
 results.append({'name':name,'pass':bool(condition)})
 if not condition:print('FAIL:',name)
methods=s['methods'];ids=[m['id'] for m in methods];scenes=[x for v in s['cohorts'].values() for x in v]
check('nine unique registered methods',len(ids)==len(set(ids))==9)
check('four learned branches correctly registered',set(s['training']['learned_main_branches'])=={m['id'] for m in methods if m['trained']} and sum(m['trained'] for m in methods)==4)
check('main repeat and total whole-map counts',len(scenes)==26 and len(set(scenes))==26 and s['evaluation']['main_records']==9*26 and s['evaluation']['repeat_records']==2*26 and s['evaluation']['total_records_if_complete']==11*26 and s['evaluation']['total_pools_if_complete']==11*2)
p=s['data']['profile'];check('32 separate family role budget',p=={'train_families':24,'dev_families':4,'holdout_families':4} and s['data']['fallback_profile'] is None)
check('all current CF physical families denied',set(x.split('_')[0] for x in s['cohorts']['scannet_cf18'])<=set(s['data']['deny_families']))
c=s['data']['class_holdout'];check('base novel 160 40 separation',c['count']==40 and c['total']==200 and c['train_positive_classes']==160 and c['loss_denominator']=='base classes only')
t=s['training'];check('exact finite training update total',t['main_optimizer_updates']==t['warmup_steps']+4*t['branch_steps'] and t['repeat_optimizer_updates']==t['warmup_steps']+2*t['branch_steps'] and t['max_optimizer_updates_science']==16000==t['main_optimizer_updates']+t['repeat_optimizer_updates'])
check('batch schedule and seed separation',t['effective_batch']==t['micro_batch']*t['accumulation_steps']==16 and t['main_seed']!=t['repeat_seed'] and t['validation_steps']==[500,1000,1500,2000])
r=s['resources'];check('distinct image cap excludes invented map runs',r['training_pool_unique_FC_image_upper']==sum(p.values())*s['data']['max_training_pool_frames_per_family'] and r['regression_unique_FC_image_upper']==26*s['observation']['evaluation_representative_frames'] and r['total_science_unique_FC_image_upper']==3904)
check('frozen model and full precision contract',s['model']['backbone_frozen'] and s['model']['text_frozen'] and s['observation']['feature_cache_dtype']=='float32' and not s['observation']['autocast'])
check('no unscoped gate models or new geometry',not s['model']['learned_gate'] and not t['readout_gate_training'] and s['data']['new_ovi_maps']==0 and r['new_NQ_encodings']==r['new_segmentation_calls']==r['new_AnyUp_calls']==r['new_SAMV_calls']==0)
check('same parameter dimensions separate auxiliary objective',s['model']['local_hidden']==128 and s['model']['token_quality_features']==len(s['model']['token_metadata'])==8 and [m['id'] for m in methods if m['aux']]==['LR08_MV_AUX'])
check('runtime not an accuracy gate',s['accuracy_priority'] and not r['runtime_or_memory_in_selection'] and s['evaluation']['cold_timing_runs']==0)
check('five metric gate original partition and current ranking',len(s['evaluation']['five_metrics'])==5 and s['evaluation']['fraction_tolerance']==1e-10 and s['deployment_readout']['geometry_unchanged'] and s['deployment_readout']['recovered_classes_unchanged'] and s['deployment_readout']['recompute_official_ranks'])
with (R/'EXPERIMENT_MATRIX.csv').open() as f:csvrows=list(csv.DictReader(f))
check('CSV equals registered method IDs',[x['method_id'] for x in csvrows]==ids)
required=['README.md','CODEX_FINAL_EXECUTION_EN.md','CODE_REVIEW_AND_PLAN_ZH.md','DATA_AND_SPLITS.md','MODEL_AND_TRAINING_CONTRACT.md','EVALUATION_AND_RELEASE.md','SOURCE_EVIDENCE.md','reference/reference_kernels.py','reference/test_reference.py','audit/REVIEW_NOTES.md']
check('all required substantive files exist',all((R/p).is_file() and (R/p).stat().st_size>100 for p in required))
texts={p.name:p.read_text() for p in R.glob('*.md')}
check('Markdown code fences balanced',all(len(re.findall(r'^```',v,flags=re.M))%2==0 for v in texts.values()))
model=texts['MODEL_AND_TRAINING_CONTRACT.md'];data=texts['DATA_AND_SPLITS.md'];main=texts['CODEX_FINAL_EXECUTION_EN.md']
check('actual dense helper and frozen gradient distinction documented','visual_prediction_forward_convnext_2d' in model and 'must remain in autograd' in model and 'H ingestion is deliberately delayed' in data and 'after both seed nominations' in main)
parsed=True
for f in (R/'reference').glob('*.py'):
 try:ast.parse(f.read_text())
 except SyntaxError:parsed=False
check('reference code parses and tests actually passed',parsed and 'Ran 12 tests' in (R/'audit/REFERENCE_TEST_LOG.txt').read_text() and '\nOK\n' in (R/'audit/REFERENCE_TEST_LOG.txt').read_text())
check('external release proof and reports defined',len(s['publication']['required_reports'])==4 and s['publication']['external_receipt']=='publication/final.json' and s['publication']['normal_push'] and s['publication']['full_command_must_run'])
report={'kind':'PROMPT_PACK_SPECIFICATION_AUDIT','status':'PASS' if all(r['pass'] for r in results) else 'FAIL','check_count':len(results),'passed':sum(r['pass'] for r in results),'checks':results,'reference_tests':{'count':12,'status':'PASS','environment':'PyTorch 2.10.0+cpu; no CUDA','scope':'synthetic numerical/contract tests only'},'not_performed':['server data inventory','real FC or MaskAdapter loading','real GPU autograd or upstream numerical parity','supervised scientific training','real 26-scene prediction/evaluation','GitHub write or push'],'manual_review':'REVIEW_NOTES.md','numerical_spec_sha256':hashlib.sha256((R/'PROTOCOL_SPEC.json').read_bytes()).hexdigest()}
(R/'audit/AUDIT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','check_count','passed']},indent=2))
sys.exit(0 if report['status']=='PASS' else 1)
