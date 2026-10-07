"""Package-only consistency audit. Does not validate server resources or science."""
from pathlib import Path
import csv, hashlib, json, re
from datetime import datetime, timezone
R=Path(__file__).resolve().parent
spec=json.loads((R/'PROTOCOL_SPEC.json').read_text())
main=(R/'CODEX_FINAL_EXECUTION_EN.md').read_text()
rows=list(csv.DictReader((R/'EXPERIMENT_MATRIX.csv').open()))
checks=[]
def ck(name, value):
    checks.append({'check':name,'passed':bool(value)})
    if not value: raise AssertionError(name)
methods=spec['methods']; ids=[x['id'] for x in methods]
scenes=[s for arr in spec['cohorts'].values() for s in arr]
ck('base revision is a complete SHA and occurs in main instruction',bool(re.fullmatch('[0-9a-f]{40}',spec['base_commit'])) and spec['base_commit'] in main)
ck('AnyUp source pin occurs in main',spec['anyup']['commit'] in main)
ck('AnyUp release SHA256/length bound',bool(re.fullmatch('[0-9a-f]{64}',spec['anyup']['checkpoint_sha256'])) and spec['anyup']['checkpoint_sha256'] in main and spec['anyup']['checkpoint_bytes']==3540612)
ck('9 unique methods',len(ids)==len(set(ids))==9)
ck('26 exact scene identities',len(scenes)==len(set(scenes))==26 and len(spec['cohorts']['replica8'])==8 and len(spec['cohorts']['scannet_cf18'])==18)
ck('pilot scenes reuse required cohort',set(spec['pilot_scenes'])<=set(scenes))
expected={(m,c,s) for m in ids for c,arr in spec['cohorts'].items() for s in arr}
actual={(x['method'],x['cohort'],x['scene']) for x in rows}
ck('234 complete matrix rows without duplicates',len(rows)==len(actual)==len(expected)==234 and actual==expected)
ck('52 inherited + 182 new rows',sum(x['role']=='inherited' for x in rows)==52 and sum(x['role']!='inherited' for x in rows)==182)
ck('18 complete scientific pools',len(ids)*len(spec['cohorts'])==spec['scope']['pooled_records']==18)
ck('matrix roles and operators match JSON',all(all(row[k]==next(m[k] for m in methods if m['id']==row['method']) for k in ['role','representation','decision','geometry_conditioning']) for row in rows))
ck('all methods appear in authoritative text',all(m in main for m in ids))
ck('no new maps/frontends/native inference/training/fits',all(spec['scope'][k]==0 for k in ['new_maps','new_frontends','new_native_NQ_inference','new_training','new_parameter_fits']))
ck('candidate/control bounds match geometry scope',spec['scope']['physical_candidate_upper_bound']==26*128 and spec['scope']['max_context_region_records']==26*128*2)
ck('AnyUp uses original single-backbone non-NATTEN checkpoint',spec['anyup']['entrypoint']=='anyup' and spec['anyup']['use_natten'] is False and spec['anyup']['precision']=='float32')
ck('fixed target-only low-support intervention',spec['constants']['trigger_hard_support_lt']==4 and spec['constants']['trigger_purity_lt']==.5 and 'Outside this set, return EXACT B1' in main)
ck('explicit margin and soft contrast constants',spec['constants']['margin_threshold_cosine']==.01 and spec['constants']['contrast_lambda']==.25 and '0.25 * eta' in main)
ck('AnyUp original visual feature/head boundary documented','BEFORE the' in main and 'visual prediction head' in main and 'UNPROJECTED' in main)
ck('full-scene occlusion, unknown-owner neutrality specified','full predicted-scene' in main and 'Unknown-owner keys are NOT treated' in main)
ck('old-cache identity misuse prevented','scientific_key` hardcodes v2 pooling' in main)
ck('available distinct from accepted','Do NOT set old `available=False`' in main)
ck('current-class official scoring not replaced','Feature reliability scores MUST NOT replace official AP confidence' in main)
ck('all five Replica metrics protected',spec['selection']['replica_nondecrease_metrics']==spec['evaluation']['metrics'])
ck('all five CF metrics compared with complete G1',spec['selection']['cf_nondecrease_vs_G1_metrics']==spec['evaluation']['metrics'])
ck('D2 is separate CF AP hurdle',spec['selection']['reference_CF_AP']=='EV00_D2' and spec['selection']['cf_APall_strictly_above_D2'])
ck('no nominal scientific regression tolerance',spec['constants']['numeric_comparison_tolerance_fraction']==1e-10)
ck('retrospective selection is not confirmation','retrospective research' in spec['selection']['type'] and 'NOT independent validation' in main)
ck('cold timing cap equals 4 x 8 x 2',spec['scope']['cold_scene_calls_max']==spec['scope']['cold_methods_max']*8*spec['scope']['cold_repeats']==64)
ck('timing comparator map only uses actual methods',all(k in ids and all(v in ids for v in vs) for k,vs in spec['timing_comparators'].items()))
ck('no hidden per-dataset switching or rescue sweep','no per-dataset switch' in main and 'unbounded rescue sweep' in main)
ck('no full test quota or fuzzing','security fuzzing' in main and 'not only toy kernels' in main)
ck('full push plus SHA verification specified',spec['branch'] in main and 'git ls-remote origin refs/heads/'+spec['branch'] in main)
ck('reference tests actually ran','Ran 8 tests' in (R/'REFERENCE_TEST_LOG.txt').read_text() and (R/'REFERENCE_TEST_LOG.txt').read_text().rstrip().endswith('OK'))
ck('all required deliverable names present',all(name in main for name in spec['publication']['reports']))
required=['README.md','CODEX_FINAL_EXECUTION_EN.md','PROTOCOL_SPEC.json','EXPERIMENT_MATRIX.csv','SOURCE_EVIDENCE.md','CODE_REVIEW_AND_PLAN_ZH.md','reference_kernels.py','test_reference_kernels.py','REFERENCE_TEST_LOG.txt','audit_package.py']
ck('all package entry files exist',all((R/f).is_file() for f in required))
files=[]
for f in sorted(R.rglob('*')):
    if f.is_file() and '__pycache__' not in f.parts and f.name!='AUDIT_REPORT.json':
        content=f.read_bytes();files.append({'path':str(f.relative_to(R)),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
audit={'status':'PASS','scope':'INSTRUCTION_PACKAGE_ONLY','date':'2026-10-07',
 'checks':checks,'check_count':len(checks),'synthetic_reference_tests':8,
 'server_experiments_executed':False,'new_FC_or_AnyUp_inference_executed':False,
 'upstream_weights_downloaded_or_run_in_this_environment':False,
 'actual_server_assets_verified':False,'remote_repository_written':False,
 'review_corrections':[
 'Bound original AnyUp checkpoint, not moving multi-backbone default.',
 'Separated v2 pooling availability from new accept/defer semantics.',
 'Included bilinear, ordinary AnyUp and margin controls.',
 'Required full selected-frame owner rasters, not partial ROI negatives.',
 'Specified post-averaged-attention reweighting and neutral-geometry test.',
 'Separated FC normalization from AnyUp ImageNet guidance.',
 'Prohibited use of fixed-v2 scientific_key for new operators.',
 'Specified deterministic timing comparator map, including no-winner path.',
 'A nonempty mask with zero area mass is an invariant error, not a hidden fallback.',
 'No per-dataset routing, no promised gain, no independent-confirmation relabeling.'
 ],'files':files}
(R/'AUDIT_REPORT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(f"PASS: {len(checks)} package checks; 8 reference kernel tests. No server experiment performed.")
