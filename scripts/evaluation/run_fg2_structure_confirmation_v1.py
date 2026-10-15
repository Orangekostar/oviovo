#!/usr/bin/env python3
"""Fresh FG2 structural controls and fixed independent confirmation pipeline."""
from pathlib import Path
import os
import sys
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
repository=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(repository/'src'),str(repository)]
from static_ovmap.structure_confirmation_v1.runner import main

if __name__=='__main__':raise SystemExit(main())
