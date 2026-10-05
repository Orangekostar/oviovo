#!/usr/bin/env python3
"""Execute the bounded prediction-preserving v2 recovery runtime experiment."""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))
sys.path.insert(0,str(REPO))

from static_ovmap.runtime_parity.experiment import main

if __name__=="__main__":
    main()
