"""Proposal backend that replays cached Proposal2D files from disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from frontend.proposal_cache import MANIFEST_FILENAME, load_proposals, read_manifest
from src.core.data_structures import Frame, Proposal2D
from src.models.proposal_backend import ProposalBackend

logger = logging.getLogger("oviovo.models.precomputed_proposal_backend")


class PrecomputedProposalBackend(ProposalBackend):
    """Load normalized Proposal2D records from a frontend cache directory."""

    def __init__(self) -> None:
        self.cache_dir: Path | None = None
        self.manifest_path: Path | None = None
        self.strict = True
        self.manifest: dict[str, Any] = {}
        self.frame_index: dict[int, dict[str, Any]] = {}
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        cache_dir_raw = str(config.get("cache_dir", "")).strip()
        manifest_raw = str(config.get("manifest_path", "")).strip()
        self.strict = bool(config.get("strict", True))

        if not cache_dir_raw and not manifest_raw:
            raise ValueError("PrecomputedProposalBackend requires cache_dir or manifest_path.")

        if manifest_raw:
            self.manifest_path = Path(manifest_raw).expanduser().resolve()
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Proposal cache manifest not found: {self.manifest_path}")
            self.cache_dir = self.manifest_path.parent
        else:
            self.cache_dir = Path(cache_dir_raw).expanduser().resolve()
            self.manifest_path = self.cache_dir / MANIFEST_FILENAME

        if self.cache_dir is None or not self.cache_dir.exists():
            raise FileNotFoundError(f"Proposal cache directory not found: {self.cache_dir}")

        if self.manifest_path.exists():
            self.manifest = read_manifest(self.manifest_path)
            self.frame_index = {
                int(entry["frame_id"]): dict(entry)
                for entry in self.manifest.get("frames", [])
            }
        else:
            if self.strict:
                raise FileNotFoundError(
                    f"Proposal cache manifest not found: {self.manifest_path}. "
                    "Either provide a valid manifest or set strict=false."
                )
            self.manifest = {}
            self.frame_index = {}

        self.last_generation_info = {
            "backend": "precomputed",
            "cache_dir": str(self.cache_dir),
            "manifest_path": str(self.manifest_path),
            "cache_frame_count": int(len(self.frame_index)),
        }

    def generate_proposals(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        frame: Frame | None = None,
    ) -> List[Proposal2D]:
        if self.cache_dir is None:
            raise RuntimeError("PrecomputedProposalBackend.initialize() must be called before inference.")
        if frame is None:
            raise ValueError("PrecomputedProposalBackend requires frame context with frame_id.")
        cache_frame_id = int(
            frame.source_frame_id
            if getattr(frame, "source_frame_id", None) is not None
            else frame.frame_id
        )

        proposals, frame_info = load_proposals(
            self.cache_dir,
            frame_id=cache_frame_id,
            expected_shape=tuple(rgb.shape[:2]),
        )

        if self.frame_index:
            indexed = self.frame_index.get(cache_frame_id)
            if indexed is None and self.strict:
                raise KeyError(
                    f"Frame {cache_frame_id} missing from manifest index {self.manifest_path}"
                )
            if indexed is not None and indexed.get("proposal_count") != len(proposals):
                logger.warning(
                    "Manifest proposal_count mismatch for frame %s: manifest=%s cache=%s",
                    cache_frame_id,
                    indexed.get("proposal_count"),
                    len(proposals),
                )

        for proposal in proposals:
            proposal.metadata = dict(proposal.metadata)
            proposal.metadata.setdefault("cache_backend", "precomputed")
            proposal.metadata.setdefault(
                "source_backend",
                frame_info.get("source_backend", proposal.backend_name),
            )

        self.last_generation_info = {
            **self.last_generation_info,
            "frame_id": int(frame.frame_id),
            "cache_frame_id": cache_frame_id,
            "proposal_count": int(len(proposals)),
            "source_backend": str(frame_info.get("source_backend", "unknown")),
            "frame_cache_file": str(frame_info.get("file", "")),
        }
        return proposals
