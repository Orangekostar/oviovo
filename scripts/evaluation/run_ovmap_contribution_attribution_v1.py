#!/usr/bin/env python3
"""Task-owned entrypoint for the frozen contribution-attribution protocol."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
from static_ovmap.contribution_attribution_v1.runner import main
if __name__=='__main__':raise SystemExit(main())
