"""New-task phases only; never invokes the parent's all/structural controller."""

from pathlib import Path
import os
import subprocess
import time

from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,read

from .binding import bind,load_binding,seal,REPO

PHASES=('bind','prepare','coarse','pilot','freeze','predict','evaluate','diagnose','select','costs','tables','publish')
SUCCESS={'bind':('BOUND_ACTUAL_PARENT',),'prepare':('EVIDENCE_COMPLETE',),'coarse':('COARSE_COMPLETE',),
    'pilot':('INTEGRATED_PILOTS_VERIFIED',),'freeze':('SOURCE_AND_CONFIG_FROZEN',),
    'predict':('PREDICTION_COMPLETE',),'evaluate':('SCIENCE_COMPLETE',),'diagnose':('DIAGNOSTICS_COMPLETE',),
    'select':('COMPLETE_TARGET_MET','COMPLETE_NO_TARGET_GAIN'),'costs':('COSTS_COMPLETE',),
    'tables':('TABLES_COMPLETE',),'publish':('PUSH_VERIFIED',)}


def source_files():
    paths=list((REPO/'src/static_ovmap/source_preserving_update').glob('*.py'))
    paths.extend([REPO/'scripts/evaluation/run_ovimap_source_preserving_update.py',
        REPO/'configs/static_ovmap/source_preserving_update_v1.json',REPO/'tests/evaluation/test_source_preserving_update.py'])
    return sorted(paths)


def freeze(binding):
    root=Path(binding['output_root']);pilot=read(root/'pilots/summary.json')
    required=['binding','evidence','coarse_worker','decisions','outputs','evaluation','analysis','selection','costs','reporting','orchestration']
    if any(not (REPO/'src/static_ovmap/source_preserving_update'/(m+'.py')).is_file() for m in required):
        raise ValueError('all phases must be implemented before full-outcome freeze')
    if pilot['status']!='INTEGRATED_PILOTS_VERIFIED':raise ValueError('two real integrated pilots required')
    acquisition=read(root/'coarse/summary.json')
    if acquisition['status']!='COARSE_COMPLETE':raise ValueError('coarse acquisition must be complete before freeze')
    representative=None
    for p in sorted((root/'coarse').glob('*/*/receipt.json')):
        frame=read(p)
        for rid,item in frame['records'].items():
            if item['available']:
                representative={'frame':str(p),'frame_identity':frame['identity'],'region_id':rid,
                    'FC_model':item['FC_model'],'feature_content_key':item['feature_content_key'],
                    'actual_worker_python':frame['actual_worker_python'],'path':'original image_tensor/signed_mask/region_vector/cosine_record'}
                break
        if representative:break
    eligible=sum(read(root/'eligibility'/(s+'.json'))['eligible_count'] for names in binding['cohorts'].values() for s in names)
    if eligible and representative is None:raise ValueError('nonempty common domain needs at least one actual successful C readout')
    index=ConsumptionIndex(root/'freeze_verifications.json');files=[index.identity(p) for p in source_files()]
    result=seal({'status':'SOURCE_AND_CONFIG_FROZEN','binding_identity':binding['identity'],
        'pilots_identity':pilot['identity'],'files':files,'before_new_full_pooled_outcomes':True,
        'actual_coarse_readout':representative,'coarse_identity':acquisition['identity'],
        'applicability':'PAIRED_UPDATES_AVAILABLE' if eligible else 'NO_APPLICABLE_PAIRED_UPDATES',
        'test_log':index.identity(root/'tests_initial.log'),
        'affected_selection_test_log':index.identity(root/'tests_selection_after_fix.log')})
    path=root/'freeze.json'
    if path.exists() and read(path)!=result:raise ValueError('frozen implementation changed; journal correction and scoped invalidation required')
    atomic_write_json(path,result);index.write_memo(root/'freeze_verifications.json');return result


def require_freeze(binding):
    root=Path(binding['output_root']);f=read(root/'freeze.json')
    if f['status']!='SOURCE_AND_CONFIG_FROZEN' or f['binding_identity']!=binding['identity']:raise ValueError('scientific freeze missing')
    index=ConsumptionIndex(root/'freeze_verifications.json')
    for item in f['files']:index.identity(item['path'],item)
    return f


def coarse(binding):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=binding['gpu'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(REPO/'src')+os.pathsep+str(REPO))
    root=Path(binding['output_root']);log=root/'coarse_execution.log';start=time.perf_counter()
    argv=[binding['FC_python'],'-m','static_ovmap.source_preserving_update.coarse_worker','--root',str(root)]
    with log.open('a') as handle:
        result=subprocess.run(argv,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,check=False)
    receipt=seal({'argv':argv,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start,'log':str(log)})
    atomic_write_json(root/'coarse/invocations'/(str(time.time_ns())+'.json'),receipt)
    if result.returncode:raise RuntimeError('FC coarse worker exited '+str(result.returncode)+'; '+str(log))
    return read(root/'coarse/summary.json')


def pilot(binding):
    from .outputs import predict
    from .evaluation import evaluate_scene
    names=binding['specification']['pilots'];root=Path(binding['output_root'])
    predictions=predict(binding,scenes=names,require_frozen=False)
    rows=[evaluate_scene((binding,s),require_frozen=False) for s in names]
    complete=predictions['status']=='PREDICTION_COMPLETE' and all(r['row_count']==9 for r in rows)
    result=seal({'status':'INTEGRATED_PILOTS_VERIFIED' if complete else 'PILOTS_INCOMPLETE',
        'scenes':{r['scene']:r['identity'] for r in rows},'predictions':predictions['identity'],
        'real_partitions_scored':sum(r['row_count'] for r in rows),'new_AnyUp_QK':0})
    atomic_write_json(root/'pilots/summary.json',result);return result


def run(args):
    root=Path(args.storage_root or args.output_root).resolve()
    phases=PHASES if args.phase=='all' else (args.phase,);results={};failures=[];binding=None
    for phase in phases:
        start=time.perf_counter()
        try:
            if phase=='bind':
                binding=bind(args.spec,args.parent_root,args.output_root,storage_root=args.storage_root,gpu=args.gpu,path_map=args.path_map)
                result=binding
            else:
                binding=binding or load_binding(root)
                if phase=='prepare':
                    from .evidence import prepare
                    result=prepare(binding)
                elif phase=='coarse':result=coarse(binding)
                elif phase=='pilot':result=pilot(binding)
                elif phase=='freeze':result=freeze(binding)
                elif phase=='predict':
                    from .outputs import predict
                    result=predict(binding)
                elif phase=='evaluate':
                    from .evaluation import evaluate_study
                    result=evaluate_study(binding)
                elif phase=='diagnose':
                    from .analysis import analyze
                    result=analyze(binding,read(root/'result_store.json'))
                elif phase=='select':
                    from .selection import select
                    result=select(read(root/'result_store.json')['pooled_metrics'],binding['specification']['selection'])
                    atomic_write_json(root/'selection.json',seal(result))
                elif phase=='costs':
                    from .costs import measure
                    result=measure(binding)
                elif phase=='tables':
                    from .reporting import report
                    result=report(binding)
                elif phase=='publish':
                    from .reporting import publish
                    result=publish(binding)
            results[phase]=result
            if result.get('status') not in SUCCESS[phase]:
                failures.append({'phase':phase,'reason':'incomplete status: '+str(result.get('status'))})
        except Exception as exc:
            failure={'phase':phase,'reason':type(exc).__name__+': '+str(exc)};failures.append(failure)
            result={'status':'FAILED','failure':failure};print('PHASE_FAILED',phase,failure['reason'],flush=True)
            if args.phase!='all':break
        root.mkdir(parents=True,exist_ok=True)
        atomic_write_json(root/'execution'/(phase+'.json'),seal({'phase':phase,'result_status':result.get('status'),
            'result_identity':result.get('identity'),'elapsed_seconds':time.perf_counter()-start,'failures':failures}))
    complete=not failures
    if args.phase=='all':
        complete=(complete and results.get('evaluate',{}).get('status')=='SCIENCE_COMPLETE'
            and results.get('publish',{}).get('status')=='PUSH_VERIFIED')
    final=seal({'status':'COMPLETE' if complete else 'INCOMPLETE','phase':args.phase,'exit_code':0 if complete else 1,
        'failures':failures,'phase_statuses':{k:v.get('status') for k,v in results.items()}})
    atomic_write_json(root/'execution'/(args.phase+'_terminal.json'),final);return final['exit_code']
