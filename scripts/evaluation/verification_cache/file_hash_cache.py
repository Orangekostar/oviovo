"""Process-local file hashing that preserves receipt mutation detection."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from time import time_ns


class StatAwareSha256:
    """Hash each unchanged file once; never trust a digest without reading it."""

    # Filesystem timestamps can share a clock tick even when exposed in ns.
    # Never reuse a digest while a same-tick write could retain its signature.
    _SETTLE_NS = 2_000_000_000

    def __init__(self, reader: Callable[[Path], str], *, max_entries: int = 65536):
        if max_entries < 1:
            raise ValueError("hash cache capacity must be positive")
        self._reader = reader
        self._max_entries = max_entries
        self._entries = OrderedDict()

    @staticmethod
    def _signature(path: Path) -> tuple[int, ...]:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def __call__(self, path: Path | str) -> str:
        path = Path(path).absolute()
        signature = self._signature(path)
        settled = time_ns() - max(signature[-2:]) >= self._SETTLE_NS
        cached = self._entries.get(path)
        if settled and cached is not None and cached[0] == signature:
            self._entries.move_to_end(path)
            return cached[1]
        digest = self._reader(path)
        if self._signature(path) != signature:
            raise RuntimeError(f"file changed while hashing: {path}")
        if not settled:
            # A recent write may be invisible to stat within the current tick.
            if self._reader(path) != digest or self._signature(path) != signature:
                raise RuntimeError(f"file changed while hashing: {path}")
            self._entries.pop(path, None)
            return digest
        self._entries[path] = (signature, digest)
        self._entries.move_to_end(path)
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return digest


def install() -> None:
    from src.static_ovmap.module_validation import assets

    if not isinstance(assets.sha256_file, StatAwareSha256):
        assets.sha256_file = StatAwareSha256(assets.sha256_file)
