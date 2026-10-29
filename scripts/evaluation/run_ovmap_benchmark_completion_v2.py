#!/usr/bin/env python3
"""Task CLI; terminal --resume reaches no numerical/model imports."""
from pathlib import Path
import sys

# The frozen parent contains both static_ovmap.* and src.static_ovmap.*
# imports. Resolve both from a direct script invocation in any directory.
repository = Path(__file__).resolve().parents[2]
for source in (repository, repository / 'src'):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from static_ovmap.benchmark_completion_v2.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
