"""Fixed-phase CLI; scientific negative results and execution failures differ."""
import argparse
from pathlib import Path

from .binding import bind,load_binding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--parent-root',required=True)
    parser.add_argument('--output-root',required=True);parser.add_argument('--storage-root')
    parser.add_argument('--path-map');parser.add_argument('--gpu');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=('bind','diagnose','screen','expand','time','report','verify','all'),default='all')
    args=parser.parse_args(argv)
    binding=bind(args.spec,args.parent_root,args.output_root,storage_root=args.storage_root,path_map=args.path_map,gpu=args.gpu)
    if args.phase=='bind':return binding
    if args.phase in ('diagnose','all'):
        from .diagnostics import diagnose
        result=diagnose(binding)
        if args.phase=='diagnose':return result
    # No empty/partial invocation may pretend the scientific branch is complete.
    from .study import run_phase
    return run_phase(binding,args.phase)
