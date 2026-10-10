"""Execute the fixed supervised learned object readout study."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'),str(REPO)]

from static_ovmap.learned_object_readout.runner import main

if __name__ == '__main__': main()
