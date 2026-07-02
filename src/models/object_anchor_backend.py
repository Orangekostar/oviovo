"""Backends for detector-driven object anchors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from src.core.data_structures import Anchor2D


class ObjectAnchorBackend(ABC):
    """Abstract backend for object-level 2D anchors."""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def generate_anchors(self, rgb: np.ndarray) -> List[Anchor2D]:
        ...


class PlaceholderObjectAnchorBackend(ObjectAnchorBackend):
    """No-op anchor backend."""

    def initialize(self, config: dict[str, Any]) -> None:
        return

    def generate_anchors(self, rgb: np.ndarray) -> List[Anchor2D]:
        return []
