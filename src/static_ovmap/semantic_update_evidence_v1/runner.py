"""Finite phase dispatch; inactive stages never launch data or GPU workers."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from .common import REPO, read, verified, write, DependencyError

def parser():
    p=argparse.ArgumentParser(description='Frozen semantic update evidence; --parent-root means STRUCT_ROOT (not FG_ROOT).')
    p.add_argument('--spec',default=str(REPO/'configs/static_ovmap/semantic_update_evidence_v1.json'))
    p.add_argument('--parent-root',default='/home/ww/crove/fg2-structure-confirmation-v1/attempt_001',help='STRUCT_ROOT: completed structure-confirmation attempt; distinct from previous task FG_ROOT.')
    p.add_argument('--output-root',required=True)
    p.add_argument('--phase',default='all')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--path-map')
    p.add_argument('--gpu',type=int)
    for name in ('scannet-root','sam2-root','sam2-python'):p.add_argument('--'+name)
    return p

def bind_or_partial(args,spec):
    from .binding import bind
    try:return bind(args,spec)
    except (DependencyError,FileNotFoundError) as error:
        write(Path(args.output_root)/'science.json',dict(status='PARTIAL_DEPENDENCY',
              error_type=type(error).__name__,error=str(error),new_head_calls=0,new_training_updates=0))
        return None

def main(argv=None):
    args=parser().parse_args(argv)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    spec=read(args.spec)
    if args.phase not in ('all','science')+tuple(spec['phases']):raise ValueError('Undefined phase')
    from .publication import terminal_resume
    if args.resume:
        final=terminal_resume(args.output_root)
        if final:
            print('PUBLISHED_TERMINAL_RESUME C2='+final['C2']+'; no numerical phases or commits launched',flush=True)
            return 0
    if args.phase=='all':
        root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True)
        if not (args.resume and (root/'execution_observed.json').exists()):
            # Construct explicit arguments instead of preserving ambiguous duplicate --phase values.
            command=[sys.executable,str(REPO/'scripts/evaluation/run_semantic_update_evidence_v1.py'),
                     '--spec',args.spec,'--parent-root',args.parent_root,'--output-root',args.output_root,'--phase','science']
            if args.resume:command.append('--resume')
            for name in ('path_map','gpu','scannet_root','sam2_root','sam2_python'):
                value=getattr(args,name)
                if value is not None:command.extend(['--'+name.replace('_','-'),str(value)])
            result=subprocess.run(command,cwd=REPO)
            write(root/'execution_observed.json',dict(status='EXIT_OBSERVED',actual_exit_code=result.returncode,command=command,
                  observer='OUTER_PROCESS_AFTER_SUBPROCESS_RUN',synthetic=False))
        binding=bind_or_partial(args,spec) or dict(output_root=args.output_root,task_specification=spec)
        from .reporting import report
        if not (args.resume and (root/'report.json').exists()):report(binding)
        from .publication import publish
        publish(binding)
        return verified(root/'execution_observed.json')['actual_exit_code']
    binding=bind_or_partial(args,spec)
    if binding is None:
        if args.phase=='report':
            from .reporting import report
            report(dict(output_root=args.output_root,task_specification=spec))
        return 2
    root=Path(binding['output_root'])
    try:
        if args.phase=='bind':return 0
        if args.phase=='report':
            from .reporting import report
            report(binding);return 0
        if args.phase=='review':
            value=verified(root/'review/primary_review.json')
            if value['status']!='COMPLETE':raise DependencyError('Primary review incomplete')
            return 0
        if args.phase=='publish':
            from .publication import publish
            publish(binding);return 0
        from .readers import import_originals
        imported=root/'A/import.json'
        if not args.resume or not imported.exists():import_originals(binding)
        if args.phase=='A_import':return 0
        from .diagnostics import run
        summary=root/'A/summary.json'
        value=verified(summary) if args.resume and summary.exists() else run(binding)
        if args.phase.startswith('A_'):return 0
        if not value['decision']['open']:
            from .stages import inactive
            inactive(root,spec,value['decision'])
            phases={name:('NOT_ACTIVATED_BY_A' if name.startswith(('B_','C_')) else 'COMPLETE')
                    for name in spec['phases'] if name not in ('report','review','publish')}
            write(root/'science.json',dict(status='COMPLETE_A_NO_RESIDUAL_SIGNAL',phases=phases,
                   decision=verified(root/'A/decision.json')['identity'],new_training_updates=0,new_workers=0,
                   new_head_calls=0,new_old_RGB_encodings=0,new_FC_images=0,new_SAM_images=0,
                   method_scope='A-only; B/C not activated by predeclared resource policy'))
            print('COMPLETE_A_NO_RESIDUAL_SIGNAL',flush=True)
            return 0
        raise DependencyError('Observed A gate changed: preserve prior transaction; activate/review the B production integration before spending its budget')
    except (DependencyError,FileNotFoundError) as error:
        write(root/'science.json',dict(status='PARTIAL_DEPENDENCY',error_type=type(error).__name__,error=str(error)))
        print(str(error),flush=True)
        return 2
