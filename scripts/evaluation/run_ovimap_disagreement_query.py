"""Execute the fixed 1009 disagreement-query protocol."""
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]

from static_ovmap.disagreement_query.runner import main

if __name__=='__main__':
    main()
