"""Opt-in verification cache for this study and its Python child processes."""

import os
import sys
from pathlib import Path

if os.environ.get("OVIMAP_VERIFY_HASH_CACHE") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from file_hash_cache import install

    install()
