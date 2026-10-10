"""Separate controller/FC/SAM2 environments; actual stage exits are observed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import contextlib
import fcntl
from .common import REPO,fixed_spec,read,verified,write
from .binding import bind

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','parent-root','output-root'):p.add_argument('--'+name,required=True)
    for name in ('storage-root','path-map','gpu','sam2-root','sam2-python'):p.add_argument('--'+name)
    p.add_argument('--data-root',action='append',default=[])
    p.add_argument('--phase',default='all',choices=fixed_spec(REPO/'configs/static_ovmap/preservation_routing_v1.json')['phases'])
    p.add_argument('--resume',action='store_true');p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    return p

def gpu_stage(binding,phase,*,sam2_root=None,sam2_python=None):
    from .data import FrozenFC,ObjectLoader,execution_config
    root=Path(binding['output_root'])
    # Share the inherited resource file; queue rather than race its nonblocking helper.
    with queued_gpu_lock(execution_config(binding)['gpu_lock']):
        fc=FrozenFC(binding);loader=ObjectLoader(fc,verified(root/'features/train-dev.json'))
        if phase=='parent_audit':
            from .parent_audit import audit
            result=audit(binding,fc,loader)
        elif phase=='engineer':
            from .engineering import engineer
            result=engineer(binding,fc,loader)
            from .integration import check
            check(binding,fc)
        elif phase=='train_R':
            from .teachers import build
            from .training import train_R
            t=build(binding,fc,loader);result=train_R(binding,fc,loader,t,17)
        elif phase=='repeat_R':
            from .training import train_R,nominate_R
            main=verified(root/'R_nomination_seed17.json')
            if main['Rstar'] is None:result=write(root/'R_nomination_seed29.json',dict(status='NOT_TRIGGERED_NO_FOUNDATION',Rstar=None))
            else:
                train_R(binding,fc,loader,verified(root/'teachers/manifest.json'),29);result=nominate_R(binding,29)
        else:
            from .transfer import gpu_dispatch
            result=gpu_dispatch(binding,phase,fc,loader,sam2_root=sam2_root,sam2_python=sam2_python)
        fc.index.write_memo(root/'features/verifications.json')
    return result

@contextlib.contextmanager
def queued_gpu_lock(path):
    with open(path,'a+') as stream:
        print('WAIT_GPU_LOCK',path,flush=True)
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            print('GPU_LOCK_ACQUIRED',flush=True);yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def main(argv=None):
    args=parser().parse_args(argv);spec=fixed_spec(args.spec)
    try:
        b=bind(args.spec,args.parent_root,args.output_root,storage_root=args.storage_root,path_map=args.path_map,
               gpu=args.gpu,data_roots=args.data_root)
    except Exception as exc:
        traceback.print_exc()
        candidate=Path(args.storage_root or args.output_root).resolve()
        mapping=read(args.path_map) if args.path_map else {}
        from .common import PathResolver
        parent=Path(PathResolver(mapping).resolve(args.parent_root)).resolve()
        if candidate!=parent and not candidate.is_relative_to(parent):
            write(candidate/'execution_last.json',dict(status='BLOCKED_DATA',phase='bind',exit_code=1,
                 error=repr(exc),argv=sys.argv if argv is None else argv,elapsed_seconds=None,
                 deployment='N0_UNCHANGED'))
        return 1
    root=Path(b['output_root']);started=time.perf_counter()
    if args._worker:
        try:
            gpu_stage(b,args.phase,sam2_root=args.sam2_root,sam2_python=args.sam2_python);return 0
        except Exception as exc:
            traceback.print_exc()
            tag=('BLOCKED_DATA' if isinstance(exc,FileNotFoundError) else 'BLOCKED_MODEL_INTERFACE' if
                 isinstance(exc,ValueError) or 'SAM2 generator' in str(exc) else 'BLOCKED_RUNTIME')
            write(root/'failures'/f'{time.time_ns()}.json',dict(status=tag,phase=args.phase,error=repr(exc),
                  worker=True,elapsed_seconds=time.perf_counter()-started))
            return 1
    phases=spec['phases'][:-1] if args.phase=='all' else [args.phase]
    receipt={};status='RUNNING';exit_code=0
    try:
        for phase in phases:
            begin=time.perf_counter();print('PHASE',phase,flush=True)
            if phase=='bind':result=b
            elif phase=='prepare':
                from .data import prepare
                result=prepare(b)
            elif phase=='select_R':
                from .selection import nominate_R
                result=nominate_R(b,17)
            elif phase in ('parent_audit','engineer','train_R','repeat_R','train_G','repeat_G','diagnose','holdout2','predict'):
                if phase in ('train_G','repeat_G','holdout2','predict') and not foundation_ready(root):
                    result=dict(status='NOT_TRIGGERED_NO_FOUNDATION')
                else:
                    cmd=[b['FC_python'],'-u',str(REPO/spec['entrypoint']),'--spec',str(Path(args.spec).resolve()),
                         '--parent-root',str(Path(args.parent_root).resolve()),'--output-root',str(root),
                         '--gpu',b['gpu'],'--phase',phase,'--resume','--_worker']
                    if args.path_map:cmd+=['--path-map',str(Path(args.path_map).resolve())]
                    for data_root in args.data_root:cmd+=['--data-root',data_root]
                    if args.sam2_root:cmd+=['--sam2-root',str(Path(args.sam2_root).resolve())]
                    if args.sam2_python:cmd+=['--sam2-python',str(Path(args.sam2_python).resolve())]
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=b['gpu'],PYTHONPATH=str(REPO)+':'+str(REPO/'src'),
                              OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    process=subprocess.run(cmd,cwd=REPO,env=env,check=False)
                    if process.returncode:
                        if args.phase=='all' and phase in ('diagnose','holdout2'):
                            result=dict(status='PARTIAL_CONFIRMATION_BLOCKED',subprocess_exit_code=process.returncode,argv=cmd)
                            exit_code=1
                        else:raise RuntimeError(f'{phase} worker exit {process.returncode}')
                    else:result=dict(status='COMPLETE',subprocess_exit_code=process.returncode,argv=cmd)
            elif phase=='lock':
                from .transfer import lock
                result=lock(b)
            elif phase=='evaluate':
                from .evaluation import run
                result=run(b)
            elif phase=='report':
                from .reporting import report
                result=report(b)
            receipt[phase]=write(root/'phases'/(phase+'.json'),dict(result,phase=phase,source_binding=b['identity'],
                  stage_elapsed_seconds=time.perf_counter()-begin))['identity']
        if args.phase=='all':
            status,exit_code=all_status(root,exit_code)
        else:status='STAGE_COMPLETE'
    except Exception as exc:
        status='BLOCKED_RUNTIME';exit_code=1;traceback.print_exc()
        write(root/'failures'/f'{time.time_ns()}.json',dict(status=status,phase=phase,error=repr(exc),
               elapsed_seconds=time.perf_counter()-started,finished_phases=receipt))
    write(root/'execution_last.json',dict(status=status,exit_code=exit_code,phase=args.phase,
                 argv=sys.argv if argv is None else argv,source_binding=b['identity'],phases=receipt,
                 elapsed_seconds=time.perf_counter()-started,deployment='N0_UNCHANGED'))
    return exit_code

def foundation_ready(root):
    return all((root/f'R_nomination_seed{s}.json').exists() and verified(root/f'R_nomination_seed{s}.json')['Rstar'] is not None for s in (17,29))

def all_status(root,exit_code):
    main=verified(root/'R_nomination_seed17.json');repeat=verified(root/'R_nomination_seed29.json')
    result=verified(root/'result_store.json')
    if exit_code or result['status']=='PARTIAL_CONFIRMATION_BLOCKED':return 'PARTIAL_CONFIRMATION_BLOCKED',1
    status=('COMPLETE_NO_2D_FOUNDATION' if main['Rstar'] is None else
            'COMPLETE_2D_NOT_REPEATED' if repeat['Rstar'] is None else 'SCIENCE_COMPLETE')
    if result['status']!=status:raise ValueError('Terminal result disagrees with the actual scientific state')
    return status,0
