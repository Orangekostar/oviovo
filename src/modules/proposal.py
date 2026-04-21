"""Module 2: Proposal Generation.

Responsibility: Produce class-agnostic 2D object proposals from RGB-D.
No semantic labeling here — purely geometric/appearance segmentation.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import Frame, Proposal2D
from src.models.proposal_backend import PlaceholderProposalBackend, ProposalBackend

logger = logging.getLogger("oviovo.modules.proposal")


class ProposalModule:
    """Generates class-agnostic 2D object proposals via a swappable backend."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.requested_backend_name = config.get("backend", "placeholder")
        self.active_backend_name = self.requested_backend_name
        self.backend_init_error: str | None = None
        self.backend_config = self._resolve_backend_config(config)
        self.backend = self._build_backend(config)
        logger.info(
            "ProposalModule initialized with requested backend=%s active backend=%s",
            self.requested_backend_name,
            self.active_backend_name,
        )

    def _create_backend(self, config: Dict[str, Any]) -> ProposalBackend:
        """Factory for proposal backends.

        TODO: Add SAM / FastSAM backend creation here.
        """
        backend_name = config.get("backend", "placeholder")
        if backend_name == "placeholder":
            return PlaceholderProposalBackend()
        elif backend_name == "sam2":
            from src.models.sam2_proposal_backend import SAM2ProposalBackend

            return SAM2ProposalBackend()
        elif backend_name == "esam":
            from src.models.esam_proposal_backend import ESAMProposalBackend

            return ESAMProposalBackend()
        elif backend_name == "entitysam":
            from src.models.entitysam_proposal_backend import EntitySAMProposalBackend

            return EntitySAMProposalBackend()
        elif backend_name == "cropformer":
            from src.models.cropformer_proposal_backend import CropFormerProposalBackend

            return CropFormerProposalBackend()
        # TODO: elif backend_name == "fastsam": return FastSAMBackend()
        else:
            logger.warning(f"Unknown backend '{backend_name}', falling back to placeholder.")
            return PlaceholderProposalBackend()

    def _resolve_backend_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge common proposal config with backend-specific nested config."""
        merged = deepcopy(config)
        backend_name = config.get("backend", "placeholder")
        backend_specific = config.get(backend_name, {})
        if isinstance(backend_specific, dict):
            merged.update(backend_specific)
        return merged

    def _build_backend(self, config: Dict[str, Any]) -> ProposalBackend:
        backend = self._create_backend(config)
        try:
            backend.initialize(self.backend_config)
            self.active_backend_name = self.requested_backend_name
            self.backend_init_error = None
            return backend
        except Exception as exc:
            if self.requested_backend_name == "placeholder":
                raise
            logger.warning(
                "Proposal backend '%s' unavailable, falling back to placeholder: %s",
                self.requested_backend_name,
                exc,
            )
            self.backend_init_error = str(exc)
            fallback_config = {**config, "backend": "placeholder"}
            self.backend_config = self._resolve_backend_config(fallback_config)
            backend = PlaceholderProposalBackend()
            backend.initialize(self.backend_config)
            self.active_backend_name = "placeholder"
            return backend

    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        """Generate proposals from an RGB-D pair.

        Args:
            rgb: (H, W, 3) uint8 image.
            depth: (H, W) float32 depth map.

        Returns:
            List of Proposal2D.
        """
        proposals = self.backend.generate_proposals(rgb, depth, frame=frame)
        # Filter by minimum area
        min_area = self.config.get("min_mask_area", 100)
        proposals = [p for p in proposals if p.area >= min_area]
        for proposal in proposals:
            proposal.metadata = dict(proposal.metadata)
            proposal.metadata["requested_backend"] = self.requested_backend_name
            proposal.metadata["actual_backend"] = self.active_backend_name
            if self.backend_init_error:
                proposal.metadata["backend_fallback_reason"] = self.backend_init_error
        logger.debug(f"Generated {len(proposals)} proposals (after area filter).")
        return proposals
