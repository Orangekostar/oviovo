#!/usr/bin/env python3
"""Run the frozen conditional semantic update experiment."""
import sys
import os
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
repo=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(repo/'src'),str(repo)]
from static_ovmap.semantic_update_evidence_v1.runner import main

if __name__=='__main__':
    raise SystemExit(main())
