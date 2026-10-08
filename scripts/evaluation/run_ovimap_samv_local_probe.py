#!/usr/bin/env python3
"""Reproduce only the prescribed four-scene local SAM-V probe."""
from pathlib import Path
import sys

task_repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(task_repo))
sys.path.insert(0,str(task_repo/'src'))

from static_ovmap.samv_local_probe.runner import main

if __name__=='__main__':
    raise SystemExit(main())
