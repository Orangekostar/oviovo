"""Immutable frontend proposal bundle for ordered pipeline scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.data_structures import Anchor2D, AnchorAssignment, Proposal2D


@dataclass(frozen=True)
class FrameProposalBundle:
    """Frontend outputs for one frame, with no map-state ownership.

    Container fields are shallow-immutable snapshots. Contained proposal and
    anchor objects remain shared references.
    """

    frame_id: int
    source_frame_id: int | None
    source_proposals: tuple[Proposal2D, ...]
    raw_proposals: tuple[Proposal2D, ...]
    anchors: tuple[Anchor2D, ...]
    anchor_assignments: tuple[AnchorAssignment, ...]
    proposal_source: str
    anchor_guided_sam_summary: dict[str, Any] = field(default_factory=dict)
    generation_timings: dict[str, float] = field(default_factory=dict)
    actual_backend: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_proposals", tuple(self.source_proposals))
        object.__setattr__(self, "raw_proposals", tuple(self.raw_proposals))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "anchor_assignments", tuple(self.anchor_assignments))
        object.__setattr__(self, "anchor_guided_sam_summary", dict(self.anchor_guided_sam_summary))
        object.__setattr__(self, "generation_timings", dict(self.generation_timings))
