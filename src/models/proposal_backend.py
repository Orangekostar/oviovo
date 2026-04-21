"""Abstract base classes for proposal generation backends.

Provides a swappable interface so the proposal module can use
SAM, FastSAM, or any future segmentation model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.core.data_structures import Frame, Proposal2D


class ProposalBackend(ABC):
    """Abstract proposal generation backend."""

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        """Produce class-agnostic 2D proposals from an RGB-D pair.

        Args:
            rgb: (H, W, 3) uint8 image.
            depth: (H, W) float32 depth map.
            frame: Optional frame context for stateful/video proposal frontends.

        Returns:
            List of Proposal2D instances.
        """
        ...


class PlaceholderProposalBackend(ProposalBackend):
    """Placeholder that generates random rectangular proposals."""

    def __init__(self) -> None:
        self.max_proposals = 5

    def initialize(self, config: dict) -> None:
        self.max_proposals = config.get("max_proposals", 5)

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        h, w = rgb.shape[:2]
        proposals = []
        n = min(self.max_proposals, 3)  # keep demo small
        for i in range(n):
            # TODO: Replace with real segmentation backend (SAM / FastSAM)
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = np.random.randint(w // 2, w)
            y2 = np.random.randint(h // 2, h)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            proposals.append(Proposal2D(
                proposal_id=i,
                mask=mask,
                bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                area=int(mask.sum()),
                confidence=np.random.uniform(0.6, 1.0),
                backend_name="placeholder",
                metadata={"source": "placeholder"},
            ))
        return proposals
