"""Actual subprocess exits precede reporting and ordinary task publication."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .common import REPO,PHASES,read,verified,write,fixed_spec,DependencyError
from .binding import preflight,load_binding
from .resume import check_consumed,phase_record,phase_unchanged

OUTPUTS={phase:phase+'.json' for phase in PHASES}
OUTPUTS.update(preflight='preflight.json',inventory='inventory.json',history='history.json',freeze='weight_lock.json')

def worker(args,spec):
    if args.phase=='preflight':return preflight(args,spec)
    root=args.output_root
    if args.phase in ('fusion_dev_predict','main_predict'):
        module=importlib.import_module('.prediction',__package__)
        return module.predict(root,'FUSION_DEV' if args.phase=='fusion_dev_predict' else 'REGRESSION')
    if args.phase in ('fusion_dev_score','main_score'):
        module=importlib.import_module('.scoring',__package__)
        return module.score(root,'FUSION_DEV' if args.phase=='fusion_dev_score' else 'REGRESSION')
    if args.phase=='freeze':
        from .selection import freeze_weights
        return freeze_weights(root)
    module=importlib.import_module('.'+{'qualitative':'visualize','nominate':'nomination'}.get(args.phase,args.phase),__package__)
    return module.run(root)

def command_args(args):
    command=[str(REPO/'scripts/evaluation/run_ovmap_contribution_attribution_v1.py'),'--spec',str(args.spec),'--output-root',str(args.output_root)]
    for option in ('parent_root','wave2_root','area_root','compact_root','backbone_root','path_map','gpu'):
        value=getattr(args,option)
        if value is not None:command+=['--'+option.replace('_','-'),str(value)]
    if args.resume:command+=['--resume']
    return command

def observed_phase(args,phase):
    root=args.output_root;folder=root/'logs';folder.mkdir(parents=True,exist_ok=True)
    output=root/OUTPUTS[phase];before=verified(output)['identity'] if output.exists() else None
    python=sys.executable;env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(REPO/'src')+':'+str(REPO))
    if phase=='surface':
        binding=load_binding(root);python=binding['FC_python'];env['CUDA_VISIBLE_DEVICES']=binding['gpu']
    command=[python,'-u',*command_args(args),'--phase',phase,'--worker']
    start=time.perf_counter()
    with (folder/(phase+'.log')).open('a') as log:
        process=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    path=root/'stage_exits.json';records=verified(path)['records'] if path.exists() else []
    after=verified(output)['identity'] if output.exists() else None
    row=dict(phase=phase,exit_code=process.returncode,command=command,observed_after_wait=True,log=str(folder/(phase+'.log')),wall_seconds=time.perf_counter()-start,
             sealed_output_before=before,sealed_output_after=after,exact_completed_output_reused=process.returncode==0 and before is not None and before==after)
    records.append(row);write(path,dict(actual_exits_only=True,records=records))
    if process.returncode==0 and (root/OUTPUTS[phase]).exists():phase_record(root,phase,root/OUTPUTS[phase])
    print('OBSERVED_PHASE_EXIT',phase,process.returncode,flush=True)
    return process.returncode

def scientific_all(args):
    code=0
    if args.resume:check_consumed(args.output_root)
    for phase in PHASES:
        if args.resume and phase_unchanged(args.output_root,phase,args.output_root/OUTPUTS[phase]):
            previous=verified(args.output_root/OUTPUTS[phase])
            if previous.get('status') in ('COMPLETE','ALL_PREDICTIONS_LOCKED','NOT_SELECTED_NO_FUSION_DEV','NOT_RUN_BY_CONTRACT'):
                print('RESUME_SEALED_PHASE',phase,flush=True);continue
        result=observed_phase(args,phase)
        if result:code=1
    return code

def coordinator(args):
    # Publication is deliberately outside the scientific child's lifetime.
    root=args.output_root
    if args.resume and (root/'publication/final.json').exists():
        from .publication import terminal_resume
        if terminal_resume(root):
            print('RESUME_PUBLICATION_VERIFIED workers=0 head_calls=0 new_commits=0',flush=True);return 0
    observed=verified(root/'execution_observed.json') if (root/'execution_observed.json').exists() else None
    reusable=args.resume and observed is not None and observed['exit_code']==0 and observed['observed_after_wait'] and all(phase_unchanged(root,p,root/OUTPUTS[p]) for p in PHASES)
    if reusable:
        check_consumed(root);code=0;print('RESUME_OBSERVED_SCIENTIFIC_CHILD workers=0',flush=True)
    else:
        command=[sys.executable,'-u',*command_args(args),'--phase','all','--scientific-all']
        start=time.perf_counter();(root/'logs').mkdir(parents=True,exist_ok=True)
        with (root/'logs/scientific_all.log').open('a') as log:
            process=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        write(root/'execution_observed.json',dict(exit_code=process.returncode,observed_after_wait=True,command=command,wall_seconds=time.perf_counter()-start,status='COMPLETE' if process.returncode==0 else 'SCIENTIFIC_CHILD_NONZERO'))
        code=process.returncode
    for phase in ('report','publish'):
        try:
            module=importlib.import_module('.'+{'report':'reporting','publish':'publication'}[phase],__package__)
            module.run(root)
        except Exception as exc:
            code=1;write(root/(phase+'_error.json'),dict(error=str(exc),traceback=traceback.format_exc()));print(phase.upper()+'_FAILED',str(exc),file=sys.stderr)
    return code

def main():
    parser=argparse.ArgumentParser(description='Frozen cached E/R/S contribution attribution; no new training/RGB/SAM/maps')
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    for option in ('parent-root','wave2-root','area-root','compact-root','backbone-root','path-map'):parser.add_argument('--'+option,type=Path)
    parser.add_argument('--gpu');parser.add_argument('--phase',choices=(*PHASES,'report','publish','all'),default='all');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);parser.add_argument('--scientific-all',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.spec=args.spec.resolve();args.output_root=args.output_root.resolve();args.output_root.mkdir(parents=True,exist_ok=True)
    spec=fixed_spec(args.spec)
    from .paths import resolve_defaults
    resolve_defaults(args,spec)
    if args.worker:
        try:
            result=worker(args,spec);print(args.phase,result['status'],flush=True)
            return 0 if result['status'] in ('COMPLETE','ALL_PREDICTIONS_LOCKED','NOT_SELECTED_NO_FUSION_DEV','NOT_RUN_BY_CONTRACT') else 1
        except Exception as exc:
            status='PARTIAL_DEPENDENCY' if isinstance(exc,(DependencyError,FileNotFoundError)) else 'FAILED_ENGINEERING'
            write(args.output_root/'errors'/(args.phase+'.json'),dict(status=status,error=str(exc),traceback=traceback.format_exc()))
            traceback.print_exc();return 1
    if args.scientific_all:return scientific_all(args)
    if args.phase=='all':return coordinator(args)
    if args.phase in ('report','publish'):
        return 0 if importlib.import_module('.'+{'report':'reporting','publish':'publication'}[args.phase],__package__).run(args.output_root) else 1
    if args.resume:check_consumed(args.output_root)
    return observed_phase(args,args.phase)
