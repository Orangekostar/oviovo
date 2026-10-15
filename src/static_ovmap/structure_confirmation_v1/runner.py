"""Explicit phase coordinator and observed GPU worker exits."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import PHASES,REPO,fixed_spec,write,verified,workload_budget,DependencyError

def parser():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--spec',type=Path,default=REPO/'configs/static_ovmap/structure_confirmation_v1.json')
    p.add_argument('--parent-root',type=Path);p.add_argument('--attribution-root',type=Path)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--path-map',type=Path);p.add_argument('--gpu')
    p.add_argument('--scannet-root',type=Path);p.add_argument('--sam2-root',type=Path);p.add_argument('--sam2-python')
    p.add_argument('--phase',choices=(*PHASES,'report','publish','all'),default='all');p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--gpu-stages',help=argparse.SUPPRESS)
    p.add_argument('--science-worker',action='store_true',help=argparse.SUPPRESS)
    return p

def gpu_worker(args):
    from .binding import load_binding
    b=load_binding(args.output_root);spec=b['task_specification']
    os.environ['CUDA_VISIBLE_DEVICES']=b['gpu']
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    from static_ovmap.learned_object_readout.features import FrozenFC,execution_config
    from static_ovmap.frozen_fc_routing.resources import queued_gpu_lock
    from .data import SparseInputs,prepare
    from . import training
    stages=(args.gpu_stages or args.phase).split(',');budget=workload_budget(args.output_root,spec)
    with queued_gpu_lock(execution_config(b)['gpu_lock']):
        fc=FrozenFC(b);loader=SparseInputs(fc,verified(args.output_root/'features/train-dev.json'))
        for phase in stages:
            if phase=='prepare_cache':value=prepare(b,fc,loader,budget)
            elif phase=='train':
                verified(args.output_root/'prepare_cache.json');value=training.train(b,fc,loader,budget)
            elif phase=='dev_select':value=training.select(b)
            elif phase=='freeze':
                value=training.freeze(b,fc)
                from .engineering import check_gradient
                check_gradient(b,fc,loader,budget)
            elif phase=='confirm_features':
                from .features import capture
                value=capture(b,fc,budget)
            elif phase in ('confirm_real_predict','confirm_gt_predict'):
                from .prediction import predict
                value=predict(b,fc,budget,domain='REAL' if phase=='confirm_real_predict' else 'GT')
            elif phase=='map_predict':
                from .mapping import predict
                value=predict(b,fc,budget)
            elif phase=='cost':
                from .cost import measure
                value=measure(b,fc,budget)
            else:raise DependencyError('GPU phase not implemented: '+phase)
            print('PHASE_COMPLETE',phase,value['identity'],flush=True)
        if fc.images and 'confirm_features' not in stages:raise ValueError('Forbidden source grid encoding during training/regression')
    return 0

def gpu_subprocess(args,phases):
    from .binding import load_binding
    b=load_binding(args.output_root)
    python=b['assets']['FC_python']
    command=[str(python),'-u',str(REPO/'scripts/evaluation/run_fg2_structure_confirmation_v1.py'),'--output-root',str(args.output_root),'--spec',str(args.spec),'--phase',phases[0],'--worker','--gpu-stages',','.join(phases)]
    started=time.time();result=subprocess.run(command,cwd=REPO,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    write(args.output_root/'execution'/('gpu_'+('_'.join(phases))+'.json'),dict(status='COMPLETE' if result.returncode==0 else 'FAILED',command=command,actual_exit_code=result.returncode,elapsed_seconds=time.time()-started))
    if result.returncode:raise DependencyError('GPU worker failed with observed exit '+str(result.returncode))

def phase(args,spec,name):
    from .binding import bind,load_binding
    if name in ('preflight','bind'):return bind(args,spec)
    if name=='train':
        from .completion import completed_training
        completed=completed_training(load_binding(args.output_root))
        if completed:
            print('PHASE_REUSED train',completed['identity'],flush=True)
            return completed
    if name in ('prepare_cache','train','dev_select','freeze','confirm_features','confirm_real_predict','confirm_gt_predict','map_predict','cost'):return gpu_subprocess(args,[name])
    module={'confirm_acquire':'acquisition','confirm_geometry':'confirmation','confirm_proposals':'confirmation','confirm_features':'confirmation','confirm_real_predict':'prediction','confirm_gt_prepare':'gt_diagnostic','confirm_gt_predict':'prediction','confirm_score':'evaluation','map_predict':'mapping','map_score':'mapping','statistics':'statistics','cost':'cost','qualitative':'qualitative','report':'reporting','publish':'publication'}[name]
    try:handler=importlib.import_module('.'+module,__package__)
    except ModuleNotFoundError as e:
        if e.name==__package__+'.'+module:raise DependencyError('Required phase implementation pending: '+name) from e
        raise
    binding=load_binding(args.output_root)
    if name=='confirm_proposals':
        from .sam_runtime import isolated_linker
        with isolated_linker(binding):return handler.run(binding,phase=name)
    return handler.run(binding,phase=name)

def main(argv=None):
    args=parser().parse_args(argv);args.output_root=args.output_root.resolve();args.output_root.mkdir(parents=True,exist_ok=True)
    if args.worker:return gpu_worker(args)
    spec=fixed_spec(args.spec)
    if args.phase!='all':phase(args,spec,args.phase);return 0
    if not args.science_worker:
        from .binding import load_binding
        if (args.output_root/'source_binding.json').exists():
            from .publication import terminal_resume
            terminal=terminal_resume(load_binding(args.output_root))
            if terminal:
                print('TERMINAL_RESUME_NO_NEW_WORK',terminal['C2'],terminal['scientific_status'],flush=True)
                return 0 if terminal['scientific_status']=='COMPLETE' else 1
        arguments=list(sys.argv[1:] if argv is None else argv)
        command=[sys.executable,'-u',str(REPO/'scripts/evaluation/run_fg2_structure_confirmation_v1.py'),*arguments,'--science-worker']
        started=time.time();result=subprocess.run(command,cwd=REPO)
        write(args.output_root/'execution_observed.json',dict(status='COMPLETE',command=command,actual_exit_code=result.returncode,elapsed_seconds=time.time()-started,observed_after_subprocess_exit=True))
        final_errors=[]
        for name in ('report','publish'):
            try:phase(args,spec,name)
            except Exception as e:final_errors.append(dict(phase=name,type=type(e).__name__,error=str(e)));print('FINAL_PHASE_FAILED',name,type(e).__name__,str(e),flush=True)
        write(args.output_root/'execution/coordinator_result.json',dict(status='COMPLETE' if result.returncode==0 and not final_errors else 'PARTIAL',science_actual_exit_code=result.returncode,final_errors=final_errors))
        return 0 if result.returncode==0 and not final_errors else 1
    errors=[]
    for name in PHASES:
        try:phase(args,spec,name)
        except Exception as e:
            errors.append(dict(phase=name,type=type(e).__name__,error=str(e)));write(args.output_root/'execution/errors.json',dict(status='PARTIAL',errors=errors));print('PHASE_FAILED',name,type(e).__name__,str(e),flush=True)
    write(args.output_root/'execution/errors.json',dict(status='PARTIAL' if errors else 'COMPLETE',errors=errors))
    write(args.output_root/'execution/final.json',dict(status='PARTIAL' if errors else 'COMPLETE',errors=errors,actual_scientific_exit_code=1 if errors else 0))
    return 1 if errors else 0
