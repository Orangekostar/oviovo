#!/usr/bin/env python3
"""The task-owned wave2 entrypoint; legacy dispatchers are never invoked."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from static_ovmap.branch_utility_wave2_v1.runner import main
if __name__=='__main__':raise SystemExit(main())
