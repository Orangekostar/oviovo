"""Noninteractive controller, observed workers and unconditional report/publish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .common import REPO, DependencyError, require, fixed_spec, verified,write,read,source_digest
from .binding import preflight,load_binding,runtime_registry

SCIENCE=('preflight','parity','diagnose','select','repeat','prefix','predict','evaluate')
GPU_PHASES=frozenset(('parity','diagnose','select','repeat','prefix','predict'))

def scientific_worker(args,spec):
    if args.phase=='preflight': return preflight(spec,args.parent_root,args.output_root,args.path_map,args.gpu)
    binding=load_binding(args.output_root); root=Path(binding['output_root'])
    if args.phase=='evaluate':
        from .evaluation import run
        return run(binding)
    import torch
    from static_ovmap.learned_object_readout.features import FrozenFC,execution_config
    from static_ovmap.frozen_fc_routing.resources import queued_gpu_lock
    from . import prediction,diagnostics,adapter
    from .common import PathResolver
    lock=Path(execution_config(binding)['gpu_lock']); require([lock.parent])
    start=time.perf_counter(); fc=None; loader=None; calls=0; status='FAILED'
    with queued_gpu_lock(lock):
        try:
            # Actual CUDA operation is the check, not an NVML warning or scan.
            probe=torch.ones(1,device='cuda'); probe.add_(1); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            fc=FrozenFC(binding); original_phi=fc.phi
            def counted_phi(value):
                nonlocal calls
                calls+=1; return original_phi(value)
            def no_rgb(*unused,**kwargs): raise RuntimeError('New RGB encoding prohibited by protocol v2')
            fc.phi=counted_phi; fc.capture=no_rgb
            heads=runtime_registry(binding,fc); features=PathResolver(binding['path_map']).rewrite(verified(Path(binding['lr_parent_root'])/'features/train-dev.json')); loader=prediction.PrefixLoader(fc,features)
            if args.phase=='diagnose': result=diagnostics.run(binding,fc,loader)
            elif args.phase=='predict': result=prediction.predict(binding,fc,heads)
            else: result=getattr(prediction,args.phase)(binding,fc,loader,heads)
            if fc.images!=0: raise ValueError('Protocol violated: new RGB images encoded')
            fc.index.write_memo(root/'features/verifications.json'); status='COMPLETE'; return result
        finally:
            import resource
            write(root/'resources'/(args.phase+'.json'),dict(phase=args.phase,status=status,wall_seconds=time.perf_counter()-start,boundary='EXISTING_CACHED_FP32_GRIDS_INCLUDES_MODEL_LOAD_AND_GPU_LOCK_WAIT',loader_cache_read_copy_and_repool_seconds=loader.seconds if loader else None,new_phi_calls=calls,new_optimizer_updates=0,new_rgb_encodings=fc.images if fc else 0,new_sam_proposals=0,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,online_FPS=None,nested_timers_not_additive=True,**adapter.STATS))

def observed_phase(args,phase):
    root=args.output_root; folder=root/'logs'; folder.mkdir(parents=True,exist_ok=True)
    binding=load_binding(root) if phase in GPU_PHASES else None
    python=binding['FC_python'] if binding else sys.executable; require([python])
    command=[python,'-u',str(REPO/'scripts/evaluation/run_fg2_surface_bias_wave1_v2.py'),'--spec',str(args.spec),'--parent-root',str(args.parent_root),'--output-root',str(root),'--phase',phase,'--worker']
    if args.gpu is not None: command+=['--gpu',args.gpu]
    if args.path_map: command+=['--path-map',str(args.path_map)]
    if args.resume: command+=['--resume']
    env=dict(os.environ)
    if binding: env.update(CUDA_VISIBLE_DEVICES=binding['gpu'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    start=time.perf_counter()
    with (folder/(phase+'.log')).open('a') as log: process=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    store=root/'stage_exits.json'; previous=verified(store)['records'] if store.exists() else []
    row=dict(phase=phase,exit_code=process.returncode,observed_by='CONTROL_SUBPROCESS_RUN_AFTER_WAIT',command=command,log=str(folder/(phase+'.log')),wall_seconds=time.perf_counter()-start,scientific_source_digest=source_digest())
    previous.append(row); write(store,dict(records=previous,actual_exits_only=True))
    if process.returncode: raise RuntimeError(f'{phase} worker returned {process.returncode}; see '+row['log'])
    return row

def controller(args,spec):
    from . import reporting,publication
    resume_error=None
    if args.phase=='all' and args.resume:
        try:
            released=publication.unchanged_release(args.output_root)
            summary_path=args.output_root/'run_summary.json'
            if released and summary_path.exists():
                summary=verified(summary_path); deps=args.output_root/'dependency_manifest.json'
                dependency=verified(deps) if deps.exists() else {}; missing=dependency.get('missing',[])
                from .common import PathResolver
                mapping=read(args.path_map) if args.path_map else {}
                requested_parent=Path(PathResolver(mapping).resolve(str(args.parent_root))).resolve()
                saved=load_binding(args.output_root) if (args.output_root/'binding.json').exists() else {}
                same_runtime=str(requested_parent)==saved.get('fg_parent_root',dependency.get('checked_parent')) and mapping==saved.get('path_map',{}) and (args.gpu is None or not saved or args.gpu==saved['gpu'])
                if same_runtime and (summary['execution_status']=='COMPLETE' or summary['execution_status']=='BLOCKED_DEPENDENCY' and any(not Path(p).exists() for p in missing)):
                    print(json.dumps(dict(resumed_verified_release=released['C2'],execution_status=summary['execution_status'],science_status=summary['science_status'],no_new_workers_or_commits=True)),flush=True)
                    return 0 if summary['execution_status']=='COMPLETE' else 1
        except (DependencyError,RuntimeError,ValueError,OSError) as exc: resume_error=exc
    error=None; status=None; code=0
    try:
        if resume_error is not None: raise resume_error
        if args.phase in ('all',*SCIENCE):
            phases=SCIENCE if args.phase=='all' else (args.phase,)
            for phase in phases: observed_phase(args,phase)
        elif args.phase=='report': reporting.run(args.output_root)
        elif args.phase=='publish': publication.run(args.output_root)
    except (DependencyError,RuntimeError,ValueError,OSError) as exc:
        code=1; error=str(exc)
        dependencies=args.output_root/'dependency_manifest.json'
        status='BLOCKED_DEPENDENCY' if isinstance(exc,DependencyError) or dependencies.exists() and verified(dependencies).get('missing') else 'PARTIAL_ENGINEERING'
        print(error,file=sys.stderr,flush=True)
        write(args.output_root/'error.json',dict(error=error,execution_status=status,traceback=traceback.format_exc()))
    finally:
        if args.phase=='all':
            try: reporting.run(args.output_root,execution_status=status,error=error)
            except Exception as exc:
                code=1; print('REPORT_FAILED',str(exc),file=sys.stderr,flush=True); write(args.output_root/'report_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
            try:
                result=publication.run(args.output_root)
                path=args.output_root/'run_summary.json'
                if path.exists():
                    summary=read(path); summary.pop('identity',None); summary.update(publication_status='PUSH_VERIFIED',publication_snapshot='VERIFIED_EXTERNAL_RECEIPT',publication=result); write(path,summary)
            except (RuntimeError,OSError,ValueError) as exc:
                code=1; print('PUBLICATION_BLOCKED',str(exc),file=sys.stderr,flush=True)
    return code

def main(argv=None):
    parser=argparse.ArgumentParser(description='Frozen FG2 raw-value calibration. parent-root means the original FG experiment; no training or RGB encoding.')
    parser.add_argument('--spec',type=Path,default=REPO/'configs/static_ovmap/surface_bias_wave1_v2.json')
    parser.add_argument('--parent-root',type=Path)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--gpu')
    parser.add_argument('--path-map',type=Path)
    parser.add_argument('--phase',choices=('all',*SCIENCE,'report','publish'),default='all')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv); spec=fixed_spec(args.spec)
    if args.parent_root is None:
        args.parent_root=next((Path(p) for p in spec['parent_root_candidates'] if (Path(p)/'source_binding.json').exists()),Path(spec['parent_root_candidates'][0]))
    args.output_root=args.output_root.resolve(); args.output_root.mkdir(parents=True,exist_ok=True)
    if args.worker:
        try:
            value=scientific_worker(args,spec); print(json.dumps(dict(phase=args.phase,status=value['status']),ensure_ascii=False),flush=True); return 0
        except Exception as exc:
            write(args.output_root/'worker_error.json',dict(phase=args.phase,error=str(exc),traceback=traceback.format_exc())); print(traceback.format_exc(),file=sys.stderr,flush=True); return 1
    return controller(args,spec)
