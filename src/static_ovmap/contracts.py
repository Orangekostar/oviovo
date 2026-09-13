"""Observation provenance; unknown measured quantities remain absent."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Observation:
    scene_id: str
    instance_id: int
    frame_id: int
    source_query_id: str
    feature_space_id: str
    feature: np.ndarray
    visible_area_px: float
    crop_bbox_xyxy: Tuple[int, int, int, int]
    source_config_hash: str
    depth_valid_ratio: Optional[float] = None
    mask_quality_proxy: Optional[float] = None
    sharpness_proxy: Optional[float] = None
    camera_direction: Optional[Tuple[float, float, float]] = None
    view_bin: Optional[int] = None
    geometry_support_frames: Optional[int] = None
    source_mask_id_or_path: Optional[str] = None

    def __post_init__(self):
        feature = np.array(self.feature, copy=True)
        if feature.ndim != 1 or not len(feature) or not np.isfinite(feature).all():
            raise ValueError('feature must be a finite nonempty vector')
        if not self.feature_space_id or not self.source_config_hash:
            raise ValueError('feature space and source config identity are required')
        if not np.isfinite(self.visible_area_px) or self.visible_area_px < 0:
            raise ValueError('visible area must be finite and nonnegative')
        for name in ('depth_valid_ratio', 'mask_quality_proxy', 'sharpness_proxy'):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(name + ' must be a measured value in [0, 1]')
        if self.camera_direction is not None:
            direction = np.asarray(self.camera_direction)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
                raise ValueError('camera direction must be a finite nonzero 3-vector')
        feature.flags.writeable = False
        object.__setattr__(self, 'feature', feature)

    @property
    def feature_dim(self):
        return len(self.feature)

    @property
    def missing_fields(self):
        return [name for name in ('depth_valid_ratio', 'mask_quality_proxy',
                'sharpness_proxy', 'camera_direction', 'view_bin',
                'geometry_support_frames', 'source_mask_id_or_path')
                if getattr(self, name) is None]
