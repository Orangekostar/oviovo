#!/usr/bin/env python3
"""Execute the pinned frozen-FC routing study."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from static_ovmap.frozen_fc_routing.runner import main
if __name__=='__main__':raise SystemExit(main())
