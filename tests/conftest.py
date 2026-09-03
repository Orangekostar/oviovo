"""Shared test-session initialization."""

from __future__ import annotations

import importlib


def _load_optional_torch_before_contract_freeze() -> None:
    # Torch decorates stdlib Enum functions at import time. Load it before the
    # T1 behavior graph is frozen so later test collection cannot alter it.
    try:
        importlib.import_module("torch")
    except (ImportError, OSError):
        pass


_load_optional_torch_before_contract_freeze()
del _load_optional_torch_before_contract_freeze
