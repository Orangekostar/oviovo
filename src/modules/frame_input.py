"""Module 1: Frame Input.

Responsibility: Load or receive RGB-D frames and package them into
a standard Frame object.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from src.core.data_structures import CameraIntrinsics, Frame

logger = logging.getLogger("oviovo.modules.frame_input")


class FrameInputModule:
    """Packages raw RGB-D data + pose into a Frame object."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.frame_counter = 0
        logger.info("FrameInputModule initialized.")

    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: CameraIntrinsics,
        timestamp: float = 0.0,
    ) -> Frame:
        """Package inputs into a Frame.

        Args:
            rgb: (H, W, 3) uint8 color image.
            depth: (H, W) float32 depth in meters.
            pose: (4, 4) camera-to-world transform.
            intrinsics: Camera intrinsic parameters.
            timestamp: Frame timestamp.

        Returns:
            A Frame dataclass instance.
        """
        frame = Frame(
            frame_id=self.frame_counter,
            rgb=rgb,
            depth=depth,
            pose=pose,
            intrinsics=intrinsics,
            timestamp=timestamp,
        )
        self.frame_counter += 1
        logger.debug(f"Frame {frame.frame_id} created: rgb={rgb.shape}, depth={depth.shape}")
        return frame
