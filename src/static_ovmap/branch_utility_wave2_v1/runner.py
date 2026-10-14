"""Observed phase exits, single-GPU native/training boundaries, honest failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .common import REPO,DependencyError,require,fixed_spec,verified,write,read,source_digest
from .binding import preflight,load_binding,runtime_registry
PHASES=('preflight','baseline','factorial','action_bank','training_gate','train','select','repeat','map_predict','map_evaluate')
GPU_PHASES={'baseline','train','select','repeat','map_predict'}
OUTPUTS=dict(preflight='preflight.json',baseline='baseline.json',factorial='factorial.json',action_bank='action_bank.json',training_gate='training_gate.json',train='training.json',select='selection.json',repeat='repeat.json',map_predict='predictions/summary.json',map_evaluate='evaluation/store.json')

def worker(args,spec):
    if args.phase=='preflight':return preflight(spec,args.parent_root,args.previous_root,args.output_root,args.path_map,args.gpu)
    binding=load_binding(args.output_root);root=Path(binding['output_root'])
    if args.phase=='factorial':
        from .capture import factorial
        from .conveyor import run
        result=factorial(root);run(binding);return result
    if args.phase=='action_bank':
        from .action_analysis import run
        return run(root)
    if args.phase=='training_gate':
        from .training import training_gate
        return training_gate(root)
    if args.phase=='map_evaluate':
        from .evaluation import run
        return run(binding)
    if args.phase=='map_predict' and not verified(root/'activation.json')['methods']:
        from .prediction import run
        return run(binding,None,None)
    if args.phase=='train' and not verified(root/'training_gate.json')['training_activated']:
        from .training import run
        return run(binding)
    import torch
    from static_ovmap.frozen_fc_routing.resources import queued_gpu_lock
    from static_ovmap.learned_object_readout.features import execution_config
    lock=Path(execution_config(binding)['gpu_lock']);require([lock.parent]);start=time.perf_counter();status='FAILED';fc=loader=None;phi_calls=phi_vectors=0;phi_seconds=0.;updates_before=0
    def updates():return sum(len(p.read_text().splitlines()) for p in (root/'training').glob('seed*/*/curve.jsonl'))
    updates_before=updates()
    with queued_gpu_lock(lock):
        try:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            probe=torch.ones(1,device='cuda');probe.add_(1);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();torch.set_num_threads(1)
            if args.phase=='train':
                from .training import run
                result=run(binding)
            elif args.phase in ('select','repeat'):
                from . import selection
                result=getattr(selection,args.phase)(root)
            else:
                from static_ovmap.learned_object_readout.features import FrozenFC
                from .prediction import PrefixLoader
                from .common import PathResolver
                fc=FrozenFC(binding);original_phi=fc.phi
                def counted_phi(value):
                    nonlocal phi_calls,phi_vectors,phi_seconds
                    phi_calls+=1;phi_vectors+=int(value.numel()/value.shape[-1]);begin=time.perf_counter();result=original_phi(value);torch.cuda.synchronize();phi_seconds+=time.perf_counter()-begin;return result
                def no_rgb(*a,**kw):raise RuntimeError('New RGB encoding prohibited by frozen wave2 protocol')
                fc.phi=counted_phi;fc.capture=no_rgb;heads=runtime_registry(binding,fc)
                if args.phase=='baseline':
                    from .capture import run
                    features=PathResolver(binding['path_map']).rewrite(verified(Path(binding['lr_parent_root'])/'features/train-dev.json'));loader=PrefixLoader(fc,features);result=run(binding,fc,loader,heads)
                else:
                    from .prediction import run
                    result=run(binding,fc,heads)
                if fc.images:raise ValueError('New RGB encodings occurred')
                fc.index.write_memo(root/'features/verifications.json')
            status='COMPLETE';return result
        finally:
            import resource
            write(root/'resources'/(args.phase+'.json'),dict(phase=args.phase,status=status,wall_seconds=time.perf_counter()-start,boundary='CACHED_FP32_GRIDS_INCLUDES_MODEL_LOAD_LOCK_WAIT_ALL_SUBSET_HEADS' if args.phase in ('baseline','map_predict') else 'CACHED24_FEATURES_ONLY_MLP_NO_FC_OR_FG2',phi_calls=phi_calls,phi_vectors=phi_vectors,phi_seconds=phi_seconds,loader_seconds=loader.seconds if loader else None,new_optimizer_updates=max(0,updates()-updates_before),new_rgb_encodings=fc.images if fc else 0,new_sam_proposals=0,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cache_budget_GiB=32,online_FPS=None,nested_timers_not_additive=True))

def observed_phase(args,phase):
    root=args.output_root;folder=root/'logs';folder.mkdir(parents=True,exist_ok=True);binding=load_binding(root) if phase in GPU_PHASES else None;python=binding['FC_python'] if binding else sys.executable;require([python])
    command=[python,'-u',str(REPO/'scripts/evaluation/run_fg2_branch_utility_wave2_v1.py'),'--spec',str(args.spec),'--parent-root',str(args.parent_root),'--previous-root',str(args.previous_root),'--output-root',str(root),'--phase',phase,'--worker']
    if args.gpu is not None:command+=['--gpu',args.gpu]
    if args.path_map:command+=['--path-map',str(args.path_map)]
    if args.resume:command+=['--resume']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if binding:env['CUDA_VISIBLE_DEVICES']=binding['gpu']
    start=time.perf_counter();source_before=source_digest()
    with (folder/(phase+'.log')).open('a') as log:process=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    path=root/'stage_exits.json';rows=verified(path)['records'] if path.exists() else [];row=dict(phase=phase,exit_code=process.returncode,observed_by='CONTROL_SUBPROCESS_RUN_AFTER_WAIT',command=command,log=str(folder/(phase+'.log')),wall_seconds=time.perf_counter()-start,source_before=source_before,source_after=source_digest());rows.append(row);write(path,dict(records=rows,actual_exits_only=True))
    if process.returncode:raise RuntimeError(f'{phase} worker returned {process.returncode}; see '+row['log'])
    return row

def validate_measured_sources(root):
    path=Path(root)/'measured_sources.json'
    if not path.exists():return
    receipt=verified(path)
    from .common import digest_file
    for relative,sha in receipt['numerical_sources'].items():
        if digest_file(REPO/relative)!=sha:raise ValueError('Measured numerical source changed; cached science cannot be republished: '+relative)
    for relative,identity in receipt['scientific_outputs'].items():
        if verified(Path(root)/relative)['identity']!=identity:raise ValueError('Sealed scientific output changed: '+relative)

def controller(args,spec):
    error=None;code=0;status='COMPLETE'
    try:
        if args.resume:validate_measured_sources(args.output_root)
        if args.phase=='all' and args.resume:
            from .publication import unchanged_release
            old=unchanged_release(args.output_root)
            if old:
                saved=load_binding(args.output_root)
                from .common import PathResolver
                mapping=read(args.path_map) if args.path_map else {}
                requested=Path(PathResolver(mapping).resolve(str(args.parent_root))).resolve()
                if str(requested)!=saved['fg_parent_root'] or str(args.previous_root.resolve())!=saved['previous_root'] or mapping!=saved['path_map'] or args.gpu is not None and args.gpu!=saved['gpu']:raise ValueError('Terminal resume runtime arguments differ from the verified release')
                print(json.dumps(dict(resumed_verified_release=old['C2'],no_new_workers_or_commits=True)));return 0
        if args.phase in ('all',*PHASES):
            for phase in PHASES if args.phase=='all' else (args.phase,):
                output=args.output_root/OUTPUTS[phase]
                if args.resume and output.exists() and verified(output).get('status') in ('COMPLETE','ACTIVATED','NOT_ACTIVATED_BY_FIXED_RULE','NOT_ACTIVATED_BY_TRAIN_GATE','EVALUATION_COMPLETE','ALL_PREDICTIONS_LOCKED'):
                    print('RESUME_SEALED_PHASE',phase,flush=True);continue
                observed_phase(args,phase)
        elif args.phase=='report':
            from .reporting import run
            run(args.output_root)
        elif args.phase=='publish':
            from .publication import run
            run(args.output_root)
    except Exception as exc:
        code=1;error=str(exc);status='BLOCKED_DEPENDENCY' if isinstance(exc,DependencyError) else 'PARTIAL_ENGINEERING';write(args.output_root/'error.json',dict(execution_status=status,error=error,traceback=traceback.format_exc()));print(error,file=sys.stderr,flush=True)
    finally:
        if args.phase=='all' and not ('old' in locals() and old):
            from . import reporting,publication
            try:reporting.run(args.output_root,execution_status=status,error=error)
            except Exception as exc:code=1;write(args.output_root/'report_error.json',dict(error=str(exc),traceback=traceback.format_exc()));print('REPORT_FAILED',exc,file=sys.stderr)
            try:publication.run(args.output_root)
            except Exception as exc:code=1;write(args.output_root/'publication_error.json',dict(error=str(exc),traceback=traceback.format_exc()));print('PUBLICATION_FAILED',exc,file=sys.stderr)
    return code

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--parent-root',type=Path,required=True);parser.add_argument('--previous-root',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--gpu');parser.add_argument('--path-map',type=Path);parser.add_argument('--phase',choices=('all',*PHASES,'report','publish'),default='all');parser.add_argument('--resume',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args();args.output_root.mkdir(parents=True,exist_ok=True);spec=fixed_spec(args.spec)
    if args.worker:
        result=worker(args,spec);print(json.dumps(dict(phase=args.phase,status=result.get('status')),ensure_ascii=False),flush=True);return 0
    return controller(args,spec)
