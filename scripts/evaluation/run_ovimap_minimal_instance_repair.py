#!/usr/bin/env python3
"""Execute the supplied minimal-instance-repair study on the actual frozen parent."""

import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = str(min(4,max(1,int(os.environ.get(name,'4')))))
repository = Path(__file__).resolve().parents[2]
for path in (repository,repository/'src'):
    sys.path.insert(0,str(path))
os.environ['PYTHONPATH'] = os.pathsep.join([str(repository/'src'),str(repository),os.environ.get('PYTHONPATH','')])

from static_ovmap.minimal_instance_repair.binding import bind,load_binding


PHASES = ('bind','observe','propose','diagnose','assets','pilot','freeze','recognize','predict',
          'evaluate','select','time','tables','publish')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=Path('configs/static_ovmap/minimal_instance_repair_v1.json'))
    parser.add_argument('--parent-root',type=Path,default=Path('/mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001'))
    parser.add_argument('--parent-reference',type=Path)
    parser.add_argument('--output-root',type=Path,default=Path('/mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001'))
    parser.add_argument('--storage-root',type=Path)
    parser.add_argument('--gpu')
    parser.add_argument('--path-map',type=Path)
    parser.add_argument('--phase',choices=(*PHASES,'all'),default='all')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    for phase in PHASES if args.phase == 'all' else (args.phase,):
        if phase == 'bind':
            bind(args.spec,args.parent_root,args.output_root,storage_root=args.storage_root,gpu=args.gpu,
                 path_map={} if args.path_map is None else json.loads(args.path_map.read_text()),
                 parent_reference=args.parent_reference)
        else:
            binding = load_binding(args.output_root)
            from static_ovmap.minimal_instance_repair.workflow import run_phase
            run_phase(binding,phase,resume=args.resume)


if __name__ == '__main__':
    main()
