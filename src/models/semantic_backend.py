"""Abstract base classes for semantic encoding backends.

Provides a swappable interface so the semantic memory module can use
SigLIP, CLIP, or any future VLM backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import List

import numpy as np


class SemanticBackend(ABC):
    """Abstract semantic encoding backend."""

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def encode_image(self, crop: np.ndarray) -> np.ndarray:
        """Encode an image crop into a feature vector.

        Args:
            crop: (H, W, 3) uint8 image crop.

        Returns:
            (D,) float32 feature vector.
        """
        ...

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string into a feature vector.

        Args:
            text: Natural language description.

        Returns:
            (D,) float32 feature vector.
        """
        ...

    @abstractmethod
    def compute_similarity(self, image_feature: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Compute similarity between an image feature and text features.

        Args:
            image_feature: (D,) image embedding.
            text_features: (K, D) text embeddings.

        Returns:
            (K,) similarity scores.
        """
        ...


class PlaceholderSemanticBackend(SemanticBackend):
    """Deterministic placeholder backend for explicit image/text matching."""

    def __init__(self) -> None:
        self.feature_dim = 512

    def initialize(self, config: dict) -> None:
        self.feature_dim = config.get("feature_dim", 512)

    def encode_image(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return self._feature_from_bytes(b"empty-image")

        crop = np.asarray(crop, dtype=np.uint8)
        stats = np.concatenate(
            [
                crop.mean(axis=(0, 1)),
                crop.std(axis=(0, 1)),
                np.array([crop.shape[0], crop.shape[1]], dtype=np.float32),
            ]
        ).astype(np.float32)
        sampled = crop[:: max(1, crop.shape[0] // 8), :: max(1, crop.shape[1] // 8), :]
        return self._feature_from_bytes(stats.tobytes() + sampled.tobytes())

    def encode_text(self, text: str) -> np.ndarray:
        return self._feature_from_bytes(text.strip().lower().encode("utf-8"))

    def compute_similarity(self, image_feature: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        return text_features @ image_feature

    def _feature_from_bytes(self, payload: bytes) -> np.ndarray:
        digest = hashlib.sha256(payload).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        feat = rng.standard_normal(self.feature_dim).astype(np.float32)
        feat /= np.linalg.norm(feat) + 1e-8
        return feat
