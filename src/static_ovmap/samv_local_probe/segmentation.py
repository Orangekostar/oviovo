"""One GPU process at a time, exact successful-leaf resume and OOM-only replanning."""

from pathlib import Path
import os
import subprocess
import time

import numpy as np

from static_ovmap.backbone_wave1.runtime import exclusive_lock

from .binding import load_scene
from .common import (REPO, ConsumptionIndex, archive, canonical_digest, read,
                     request_key, verified, write)


def dependency(query,model,assets,index):
    workers=[index.identity(Path(__file__).with_name(name)) for name in
             (model.lower()+'_worker.py','worker_io.py')]
    return request_key(query,model,canonical_digest({'assets':assets['identity'],'worker':workers}))


def run_batch(binding,assets,model,queries,*,kind='SCIENCE',destinations=None,force_timing=False):
    if model not in ('SAMV','SAM2'):raise ValueError('unknown segmentor')
    root=Path(binding['output_root']);index=ConsumptionIndex(root/'segmentation/verifications.json');tasks=[];rows=[]
    for pos,query in enumerate(queries):
        dest=Path(destinations[pos]) if destinations else root/'segmentation'/query['scene']/model/str(query['owner'])
        key=dependency(query,model,assets,index)
        for frame in query['frames']:index.identity(frame['canonical']['file']['path'],frame['canonical']['file'])
        path=dest/'receipt.json'
        if path.exists() and not force_timing:
            previous=verified(path)
            if previous.get('input_identity')!=key:
                raise ValueError('segmentor dependency changed; explicitly invalidate its affected descendants')
            if previous['status']=='COMPLETE':
                index.identity(previous['raw_arrays']['path'],previous['raw_arrays']);rows.append(previous);continue
            retry_path=dest/'identical_retry.json'
            if retry_path.exists() and verified(retry_path)['input_identity']==key:
                raise RuntimeError('one identical failed-leaf retry already used: '+str(dest))
            write(retry_path,{'input_identity':key,'identical_retry_number':1,'failed_receipt':previous['identity']})
            archive(root,[path],{'phase':'segment','unchanged_failed_attempt_retried':True})
        if force_timing and path.exists():raise ValueError('timing leaves are single use; resume completed measurement instead')
        tasks.append({'query':query,'dest':str(dest),'kind':kind,'input_identity':key})
    if not tasks:return rows
    invocation=str(time.time_ns());folder=root/'segmentation/invocations'/invocation
    folder.mkdir(parents=True,exist_ok=True);request=folder/'request.json';receipt=folder/'receipt.json'
    write(request,{'assets':assets,'specification':binding['specification'],'tasks':tasks,'receipt':str(receipt)})
    python=assets['environments'][model.lower()]
    if not Path(python).is_file():raise FileNotFoundError(f'isolated {model} environment is not ready: {python}')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=binding['gpu'],PYTHONPATH=str(REPO/'src'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    # The controller prepends its own CUDA libraries while loading old geometry.
    # Let each isolated torch wheel resolve its matching CUDA libraries by RPATH.
    inherited_library_path=env.pop('LD_LIBRARY_PATH',None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    argv=[python,'-m','static_ovmap.samv_local_probe.'+model.lower()+'_worker','--request',str(request)]
    lock=binding['assets']['execution_config']['gpu_lock']
    if str(binding['gpu'])!=str(binding['assets']['execution_config']['gpu']):
        lock=str(Path(lock).with_name('.visual-gpu-'+binding['gpu']+'.lock'))
    begin=time.perf_counter()
    with exclusive_lock(Path(lock)),(folder/'worker.log').open('w') as log:
        process=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
    result=write(folder/'process.json',{'argv':argv,'exit_code':process.returncode,
        'wall_seconds':time.perf_counter()-begin,'log':str(folder/'worker.log'),'GPU_lock':lock,
        'planned_queries':len(tasks),'kind':kind,
        'worker_library_isolation':{'LD_LIBRARY_PATH_removed':inherited_library_path is not None,
                                    'CUDA_resolution':'isolated_torch_wheel_RPATH'}})
    if not receipt.exists():raise RuntimeError(f'{model} worker exited {process.returncode} without receipt: {folder}/worker.log')
    completed=verified(receipt)
    if process.returncode!=0 or completed['status']!='COMPLETE':
        error=RuntimeError(f'{model} {completed["status"]}: {folder}/worker.log')
        error.worker_status=completed['status'];error.worker_receipt=completed;raise error
    for row in completed['rows']:
        index.identity(row['raw_arrays']['path'],row['raw_arrays']);rows.append(row)
    index.write_memo(root/'segmentation/verifications.json');return rows


def pilot_queries(binding):
    root=Path(binding['output_root']);result=[]
    for cohort,names in binding['cohorts'].items():
        for scene in names:
            plan=verified(root/'query_plan'/scene/'receipt.json')
            if plan['queries']:
                result.append(plan['queries'][0]);break
    return result


def pilot(binding,assets):
    from .query_plan import plan
    from .lifting import count_view_votes,arbitrate
    from .outputs import build
    from .query_plan import load_observation
    root=Path(binding['output_root']);profile=read(root/'resource_profile.json') if (root/'resource_profile.json').exists() else {'window_length':6,'OOM_fallback_used':False}
    pilots=pilot_queries(binding);rows=[]
    try:
        for model in ('SAM2','SAMV'):
            rows.extend(run_batch(binding,assets,model,pilots,kind='ENGINEERING_PILOT'))
    except RuntimeError as exc:
        if getattr(exc,'worker_status',None)!='CUDA_OOM' or profile['OOM_fallback_used']:raise
        archive(root,[root/'query_plan',root/'segmentation',root/'pilots'],
            {'reason':'GENUINE_CUDA_OOM_GLOBAL_SIX_TO_FOUR','worker':exc.worker_receipt['identity']})
        write(root/'resource_profile.json',{'window_length':4,'OOM_fallback_used':True,
            'trigger_status':'CUDA_OOM','before_any_new_AP':True})
        plan(binding,count=4);return pilot(binding,assets)
    checks=[]
    for query in pilots:
        inputs=load_scene(binding,query['scene']);plan=verified(root/'query_plan'/query['scene']/'receipt.json')
        with np.load(plan['domain']['path'],allow_pickle=False) as arrays:
            editable=arrays['editable'];domain=arrays[f'owner_{query["owner"]}']
        for model in ('SAM2','SAMV'):
            row=next(r for r in rows if r['scene']==query['scene'] and r['owner']==query['owner'] and r['model']==model)
            masks=[];sources=[];valids=[]
            with np.load(row['raw_arrays']['path'],allow_pickle=False) as arrays:
                for slot,frame in enumerate(query['frames']):
                    source,valid=load_observation(frame,len(inputs.xyz));sources.append(source);valids.append(valid)
                    masks.append(arrays[f'mask_{slot}'])
            n,k=count_view_votes(sources,valids,masks,len(inputs.xyz))
            if not row['outcome']['anchor_adherent']:n[:]=0;k[:]=0
            prompts={int(r):query['owner'] for r in query['prompt_source_rows']}
            owners,audit=arbitrate(inputs.g1.owner_ids,[query['owner']],{query['owner']:(n,k)},
                {query['owner']:domain},editable,prompts)
            payload,out_audit=build(inputs,'ENGINEERING_'+model,owners,{},editable,prompts)
            checks.append({'scene':query['scene'],'owner':query['owner'],'model':model,
                'segmentation_identity':row['identity'],'real_array_lifter_validated':True,
                'actual_output_invariants':out_audit,'arbitration':audit,
                'prediction_key':payload.prediction_key,'new_AP_scored':False})
    status='REAL_ENGINEERING_PILOTS_VERIFIED' if len(pilots)==len(binding['cohorts']) else 'NO_QUERIABLE_TARGET_IN_COHORT'
    result=write(root/'pilots/summary.json',{'status':status,'checks':checks,
        'pilot_queries':[{'scene':q['scene'],'owner':q['owner'],'identity':q['identity']} for q in pilots],
        'cohorts_without_query':[c for c,names in binding['cohorts'].items() if not any(q['scene'] in names for q in pilots)],
        'profile':profile,'new_AP_scored':False,'raw_masks_and_shared_JPEGs_actual':True})
    if not (root/'resource_profile.json').exists():write(root/'resource_profile.json',profile)
    return result


def segment(binding,assets):
    from .runner import require_freeze
    require_freeze(binding);root=Path(binding['output_root']);queries=[]
    for scene in binding['scenes']:queries+=verified(root/'query_plan'/scene/'receipt.json')['queries']
    methods={model:run_batch(binding,assets,model,queries) for model in ('SAM2','SAMV')}
    return write(root/'segmentation/summary.json',{'status':'SEGMENTATION_COMPLETE',
        'targets':len(queries),'methods':{m:[r['identity'] for r in rows] for m,rows in methods.items()},
        'effective_profile':verified(root/'resource_profile.json'),'model_failures_as_zero_masks':False})
