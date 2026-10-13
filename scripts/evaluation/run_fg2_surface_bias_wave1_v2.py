#!/usr/bin/env python3
"""Entry point for the isolated, fixed-protocol FG2 calibration study."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from static_ovmap.surface_bias_wave1_v2.runner import main
if __name__=='__main__':
    raise SystemExit(main())
