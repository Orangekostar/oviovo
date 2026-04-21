"""Module 8: Semantic Memory (Layer 7).

Responsibility: Maintain object-level semantic memory.
Semantics start ONLY after instance stabilization.
No dense per-point language features — only object-level embeddings.
Semantics must not drive low-level patch association (Rule C).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from src.core.data_structures import ObjectMap, ObjectState, SystemState
from src.models.semantic_backend import PlaceholderSemanticBackend, SemanticBackend

logger = logging.getLogger("oviovo.modules.semantic_memory")


class SemanticMemoryModule:
    """Object-level semantic memory with swappable VLM backend.

    Semantic update happens ONLY after instance stabilization:
      - obj.update_count >= min_stable_observations
      - obj.state == ACTIVE

    Uses instance-level feature aggregation and text matching.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.backend: SemanticBackend = self._create_backend(config)
        self.backend.initialize(config)
        self.top_k = config.get("top_k_labels", 5)
        self.confidence_decay = config.get("confidence_decay", 0.95)
        self.min_obs = config.get("min_observations_for_label", 3)
        self.min_stable_observations = config.get("min_stable_observations", 3)
        self.min_stability_score = config.get("min_stability_score", 0.2)
        self.label_candidates = list(config.get(
            "label_candidates",
            ["chair", "table", "cabinet", "sofa", "bed", "plant", "monitor", "lamp"],
        ))
        self.text_features = self._build_text_feature_bank(self.label_candidates)
        self.last_updated_object_ids: List[int] = []
        logger.info(f"SemanticMemoryModule initialized with backend: {config.get('backend', 'placeholder')}")

    def _create_backend(self, config: Dict[str, Any]) -> SemanticBackend:
        """Factory for semantic backends."""
        backend_name = config.get("backend", "placeholder")
        if backend_name == "placeholder":
            return PlaceholderSemanticBackend()
        logger.warning(f"Unknown backend '{backend_name}', falling back to placeholder.")
        return PlaceholderSemanticBackend()

    def _build_text_feature_bank(self, labels: List[str]) -> np.ndarray:
        if not labels:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self.backend.encode_text(label) for label in labels], axis=0)

    def process(
        self,
        state: SystemState,
        rgb: np.ndarray,
        updated_object_ids: List[int],
    ) -> SystemState:
        """Update semantic memory for recently associated objects.

        Gated on instance stability: only update objects that are
        sufficiently stable (enough observations, ACTIVE state).

        Args:
            state: Current system state.
            rgb: (H, W, 3) current frame RGB.
            updated_object_ids: IDs of objects that were updated this frame.

        Returns:
            Updated system state.
        """
        self.last_updated_object_ids = []
        for obj_id in updated_object_ids:
            obj = state.objects.get(obj_id)
            if obj is None:
                continue

            stability_score = float(
                obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)
            )

            # Gate: semantics only after instance stabilization
            if obj.state != ObjectState.ACTIVE:
                obj.semantic_memory.debug["stability_gate"] = {
                    "passed": False,
                    "reason": "object_not_active",
                    "update_count": int(obj.update_count),
                    "stability_score": stability_score,
                }
                continue
            if obj.update_count < self.min_stable_observations:
                obj.semantic_memory.debug["stability_gate"] = {
                    "passed": False,
                    "reason": "insufficient_observations",
                    "update_count": int(obj.update_count),
                    "stability_score": stability_score,
                }
                logger.debug(
                    f"Object {obj_id}: skipping semantic update "
                    f"(update_count={obj.update_count} < {self.min_stable_observations})"
                )
                continue
            if stability_score < self.min_stability_score:
                obj.semantic_memory.debug["stability_gate"] = {
                    "passed": False,
                    "reason": "insufficient_tsdf_stability",
                    "update_count": int(obj.update_count),
                    "stability_score": stability_score,
                }
                continue

            self._update_semantic(obj, rgb)
            self.last_updated_object_ids.append(obj_id)

        return state

    def _update_semantic(self, obj: ObjectMap, rgb: np.ndarray) -> None:
        """Update semantic memory for a single stabilized object.

        Uses object-centric view selection and instance-level feature aggregation.
        """
        mem = obj.semantic_memory

        crop, bbox_xyxy, view_selection_score = self._select_object_view(obj, rgb)
        feature = self.backend.encode_image(crop)

        # Add to feature bank
        mem.feature_bank.append(feature)
        mem.observation_count += 1

        # Instance-level feature aggregation (weighted mean pooling)
        mem.aggregated_feature = np.mean(mem.feature_bank, axis=0)
        norm = np.linalg.norm(mem.aggregated_feature)
        if norm > 1e-8:
            mem.aggregated_feature /= norm

        similarities = self._match_candidate_labels(mem.aggregated_feature)
        ranked_indices = np.argsort(similarities)[::-1]
        top_indices = ranked_indices[: self.top_k]
        if mem.observation_count >= self.min_obs:
            mem.label_hypotheses = [
                (self.label_candidates[idx], float(similarities[idx]))
                for idx in top_indices
            ]

        mem.confidence_history.append(float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0)
        mem.debug = {
            "stability_gate": {
                "passed": True,
                "reason": "stable_instance",
                "update_count": int(obj.update_count),
                "stability_score": float(
                    obj.debug.get("global_instance_substrate", {}).get("stability_score", 0.0)
                ),
            },
            "selected_bbox_xyxy": bbox_xyxy,
            "view_selection_score": float(view_selection_score),
            "candidate_labels": list(self.label_candidates),
            "similarity_scores": [float(score) for score in similarities.tolist()],
        }

        logger.debug(
            f"Object {obj.object_id}: semantic updated, "
            f"{mem.observation_count} observations, "
            f"{len(mem.feature_bank)} features in bank."
        )

    def _select_object_view(self, obj: ObjectMap, rgb: np.ndarray) -> tuple[np.ndarray, List[int], float]:
        """Choose an object-centric crop from recent observations."""
        best_bbox: np.ndarray | None = None
        best_score = -1.0
        h, w = rgb.shape[:2]

        for observation in obj.observations[-5:]:
            bbox = observation.crop_bbox
            if bbox is None:
                continue
            bbox = np.asarray(bbox, dtype=np.float32)
            x1 = int(np.clip(np.floor(bbox[0]), 0, max(w - 1, 0)))
            y1 = int(np.clip(np.floor(bbox[1]), 0, max(h - 1, 0)))
            x2 = int(np.clip(np.ceil(bbox[2]), x1 + 1, w))
            y2 = int(np.clip(np.ceil(bbox[3]), y1 + 1, h))
            area = max((x2 - x1) * (y2 - y1), 1)
            area_score = min(1.0, area / max(h * w * 0.25, 1))
            geometry_score = min(1.0, len(observation.patch.points) / 128.0)
            score = 0.6 * area_score + 0.4 * geometry_score
            if score > best_score:
                best_score = score
                best_bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

        if best_bbox is None:
            side = min(h, w, 32)
            x1 = max((w - side) // 2, 0)
            y1 = max((h - side) // 2, 0)
            best_bbox = np.array([x1, y1, x1 + side, y1 + side], dtype=np.int32)
            best_score = 0.0

        x1, y1, x2, y2 = best_bbox.tolist()
        crop = rgb[y1:y2, x1:x2]
        return crop, [int(x1), int(y1), int(x2), int(y2)], float(best_score)

    def _match_candidate_labels(self, feature: np.ndarray) -> np.ndarray:
        """Explicit instance-level text matching."""
        if self.text_features.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.backend.compute_similarity(feature, self.text_features).astype(np.float32)
