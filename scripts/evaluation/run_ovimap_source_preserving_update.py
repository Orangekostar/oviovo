#!/usr/bin/env python3
"""Execute the prescribed source-preserving update study."""
import argparse
from pathlib import Path
import os
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='4'

from static_ovmap.source_preserving_update.orchestration import run,PHASES

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',default='configs/static_ovmap/source_preserving_update_v1.json')
    parser.add_argument('--parent-root',default='/mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001')
    parser.add_argument('--output-root',default='/mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001')
    parser.add_argument('--storage-root');parser.add_argument('--path-map');parser.add_argument('--gpu')
    parser.add_argument('--phase',choices=(*PHASES,'all'),default='all');parser.add_argument('--resume',action='store_true')
    raise SystemExit(run(parser.parse_args()))
