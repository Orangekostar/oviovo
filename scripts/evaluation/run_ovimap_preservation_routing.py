"""Run the fixed preservation/routing protocol."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
from static_ovmap.preservation_routing.runner import main
if __name__=='__main__':sys.exit(main())
