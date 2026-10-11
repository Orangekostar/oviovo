"""Task-owned phase graph with actual worker exits and preserved attempts."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from .common import REPO,fixed_spec,read,verified,write
from .binding import bind

GPU_PHASES=('engineer','diagnostic_D1','train','repeat','diagnose','predict','confirm')

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','parent-root','output-root'):p.add_argument('--'+name,required=True)
    for name in ('storage-root','path-map','gpu','sam2-root','sam2-python'):p.add_argument('--'+name)
    p.add_argument('--data-root',action='append',default=[])
    p.add_argument('--phase',default='all',choices=fixed_spec(REPO/'configs/static_ovmap/frozen_fc_routing_v1.json')['phases'])
    p.add_argument('--resume',action='store_true');p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    return p

def gpu_stage(binding,phase,*,sam2_root=None,sam2_python=None):
    from .data import FrozenFC,ObjectLoader,CachedObjectLoader,execution_config
    from .resources import queued_gpu_lock
    import torch
    root=Path(binding['output_root'])
    with queued_gpu_lock(execution_config(binding)['gpu_lock']):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=binding['gpu'] or torch.cuda.device_count()!=1:raise ValueError('Exactly one bound GPU required')
        if 'A40' not in torch.cuda.get_device_name(0):raise ValueError('Fixed A40 device required')
        fc=FrozenFC(binding);loader_type=CachedObjectLoader if phase in ('train','repeat','diagnose') else ObjectLoader
        loader=loader_type(fc,verified(root/'features/train-dev.json'))
        if phase=='engineer':
            from .engineering import engineer,sparse_parity
            from .integration import check
            r=engineer(binding,fc,loader);loader.cache.clear();loader.bytes=0;s=check(binding,fc)
            parity=sparse_parity(binding,fc,loader)
            result=dict(status='COMPLETE',engineering=r['identity'],two_scene=s['identity'],sparse_loader_parity=parity['identity'])
        elif phase=='diagnostic_D1':
            from .diagnostics import diagnostic_D1
            result=diagnostic_D1(binding,fc,loader)
        elif phase in ('train','repeat'):
            from .training import train
            from .selection import nominate
            result=train(binding,fc,loader,17 if phase=='train' else 29)
            if phase=='repeat' and result['status']=='COMPLETE':result=nominate(binding,29)
        elif phase=='diagnose':
            from .transfer import diagnose
            result=diagnose(binding,fc,loader)
        elif phase=='predict':
            from .prediction import run
            result=run(binding,fc)
        else:
            from .confirmation import confirm
            result=confirm(binding,fc,sam2_root=sam2_root,sam2_python=sam2_python)
        fc.index.write_memo(root/'features/verifications.json')
        return dict(result,worker_cost=dict(physical_image_encodings=fc.images,loader_workload=loader.workload,
                    sparse_hits=getattr(loader,'hits',0),sparse_misses=getattr(loader,'misses',0),
                    GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False))

def inactive(binding,phase):
    root=Path(binding['output_root'])
    if phase=='repeat' and verified(root/'nomination_seed17.json')['nominee'] is None:
        return write(root/'nomination_seed29.json',dict(status='NOT_TRIGGERED_NO_FG_SIGNAL',seed=29,nominee=None,
            same_nominee_qualified=False,any_qualified=False,checkpoints={},arms={},qualified={}))
    if phase in ('predict','confirm') and not verified(root/'predictor_lock.json')['activated_full_path']:
        value=dict(status='NOT_TRIGGERED_BY_DEV_REPEAT',scientific_scene_rows=0,physical_image_encodings=0)
        return write(root/('predictions/summary.json' if phase=='predict' else 'confirmation/receipt.json'),value)
    return None

def stage(binding,phase,args):
    root=Path(binding['output_root']);skipped=inactive(binding,phase)
    if skipped is not None:return skipped
    if phase=='bind':return binding
    if phase=='prepare':
        from .data import prepare
        return prepare(binding)
    if phase=='prior_audit':
        from .prior_audit import audit
        return audit(binding)
    if phase=='nominate':
        from .selection import nominate
        return nominate(binding,17)
    if phase=='lock':
        from .transfer import lock
        return lock(binding)
    if phase=='evaluate':
        from .evaluation import run
        return run(binding)
    if phase=='report':
        from .reporting import report
        return report(binding)
    if phase not in GPU_PHASES:raise ValueError('Unknown phase '+phase)
    cmd=[binding['FC_python'],'-u',str(REPO/binding['specification']['entrypoint']),'--spec',str(Path(args.spec).resolve()),
        '--parent-root',str(Path(args.parent_root).resolve()),'--output-root',str(root),'--gpu',binding['gpu'],
        '--phase',phase,'--resume','--_worker']
    if args.path_map:cmd+=['--path-map',str(Path(args.path_map).resolve())]
    for p in args.data_root:cmd+=['--data-root',p]
    if args.sam2_root:cmd+=['--sam2-root',args.sam2_root]
    if args.sam2_python:cmd+=['--sam2-python',args.sam2_python]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=binding['gpu'],PYTHONPATH=str(REPO)+':'+str(REPO/'src'),
                                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    process=subprocess.run(cmd,cwd=REPO,env=env)
    if process.returncode:raise RuntimeError(f'{phase} worker exit {process.returncode}')
    return verified(root/'stage_results'/(phase+'.json'))

def main(argv=None):
    args=parser().parse_args(argv);started=time.perf_counter();attempt=str(time.time_ns());root=Path(args.storage_root or args.output_root).resolve()
    try:
        b=bind(args.spec,args.parent_root,args.output_root,storage_root=args.storage_root,path_map=args.path_map,
               gpu=args.gpu,data_roots=args.data_root)
    except Exception as exc:
        traceback.print_exc();root.mkdir(parents=True,exist_ok=True)
        write(root/'failures'/(attempt+'.json'),dict(status='BLOCKED_DEPENDENCY_OR_RUNTIME',phase='bind',error=repr(exc)))
        return 1
    if args._worker:
        try:
            result=gpu_stage(b,args.phase,sam2_root=args.sam2_root,sam2_python=args.sam2_python)
            write(root/'stage_results'/(args.phase+'.json'),dict(result,source_binding=b['identity']))
            return 0
        except Exception as exc:
            traceback.print_exc();write(root/'failures'/(attempt+'.json'),dict(status='PARTIAL_ACTIVE_STAGE',phase=args.phase,
                  error=repr(exc),worker=True,elapsed_seconds=time.perf_counter()-started));return 1
    phases=b['specification']['phases'][:-1] if args.phase=='all' else [args.phase]
    finished={};failed=[];exit_code=0;status='STAGE_COMPLETE'
    for phase in phases:
        begin=time.perf_counter();print('PHASE',phase,flush=True)
        try:result=stage(b,phase,args)
        except Exception as exc:
            traceback.print_exc();failed.append(phase);exit_code=1
            result=dict(status='PARTIAL_ACTIVE_STAGE',error=repr(exc))
            write(root/'failures'/f'{attempt}_{phase}.json',dict(result,phase=phase,worker=False))
        result=dict(result,phase=phase,attempt=attempt,source_binding=b['identity'],stage_elapsed_seconds=time.perf_counter()-begin)
        p=root/'phases'/(phase+'.json')
        if p.exists():
            dest=root/'phase_history'/attempt/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        finished[phase]=write(p,result)['identity']
        if failed and phase not in ('diagnose','predict','evaluate','confirm','report'):
            if args.phase=='all':
                try:stage(b,'report',args)
                except Exception:traceback.print_exc()
            break
    if args.phase=='all':
        if exit_code:status='PARTIAL_ACTIVE_STAGE'
        else:
            result=verified(root/'result_store.json');status=result['status']
            if status in ('PARTIAL_ACTIVE_STAGE','BLOCKED_DEPENDENCY_OR_RUNTIME'):exit_code=1
    path=root/'execution_last.json'
    if path.exists():
        p=root/'execution_history'/(attempt+'.json');p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,p)
    write(path,dict(status=status,exit_code=exit_code,phase=args.phase,attempt=attempt,
        argv=sys.argv if argv is None else argv,source_binding=b['identity'],phases=finished,failed_phases=failed,
        elapsed_seconds=time.perf_counter()-started,deployment='N0_UNCHANGED'))
    return exit_code
