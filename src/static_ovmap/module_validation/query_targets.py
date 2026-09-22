"""Training-only current-frame annotation projection, separate from Q ranking."""

import numpy as np

from .query_gain_policy import identifiable_target, nll_gain_label
from .static_regions import project_visible_rows


class CurrentTargetBuilder:
    def __init__(self, annotations, class_ids):
        self.annotations = annotations
        self.class_ids = tuple(map(int, class_ids))
        self.frame_id, self.raster = None, None
        self.events = []

    def __call__(self, event, candidate, current):
        frame = current["frame"]
        if self.frame_id != frame["frame_id"]:
            rows, pixels = project_visible_rows(self.annotations["xyz"], frame["pose_c2w"],
                                               frame["intrinsics"], current["depth"])
            self.raster = np.zeros(current["depth"].shape, np.int64)
            self.raster.reshape(-1)[pixels] = self.annotations["gt_semantic"][rows]
            self.frame_id = frame["frame_id"]
        labels = self.raster[candidate.global_mask]
        target = identifiable_target(labels, allowed_class_ids=self.class_ids)
        gain = None
        if target is not None:
            after = event["after_scores"]
            gain = 0. if after is None else nll_gain_label(event["before_scores"], after,
                                                          target_index=self.class_ids.index(target))
        self.events.append({"scene_id": event["scene_id"], "request_id": event["request_id"],
            "frame_id": event["frame_id"], "success": event["success"], "target_class": target,
            "valid_annotated_pixels": int(np.count_nonzero(labels > 0)), "gain_target": gain})
