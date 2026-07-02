"""YOLO-World based 2D anchor backend."""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, List

import numpy as np
from PIL import Image

from src.core.data_structures import Anchor2D
from src.models.object_anchor_backend import ObjectAnchorBackend

logger = logging.getLogger("oviovo.models.yoloworld_anchor_backend")


class YOLOWorldAnchorBackend(ObjectAnchorBackend):
    """Generate detector anchors using a YOLO-World checkpoint."""

    def __init__(self) -> None:
        self.model = None
        self.external_python = ""
        self.helper_script = ""
        self.model_path = ""
        self.device = "cuda:0"
        self.classes: list[str] = []
        self.confidence_threshold = 0.2
        self.max_detections = 128
        self.helper_max_retries = 3
        self.helper_retry_sleep_sec = 0.5
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any]) -> None:
        self.model_path = str(config.get("model_path", "/home/phl/vv/yolov8s-world.pt")).strip()
        if not self.model_path:
            raise ValueError("YOLO-World model_path is empty.")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO-World checkpoint not found: {self.model_path}")

        self.device = str(config.get("device", "cuda:0")).strip() or "cuda:0"
        self.external_python = str(config.get("python_executable", "")).strip()
        helper_default = Path(__file__).resolve().parents[2] / "frontend" / "yoloworld_anchor_helper.py"
        self.helper_script = str(config.get("helper_script", helper_default)).strip()
        self.classes = [str(item).strip() for item in config.get("classes", []) if str(item).strip()]
        self.confidence_threshold = float(config.get("confidence_threshold", 0.2))
        self.max_detections = int(config.get("max_detections", 128))
        self.helper_max_retries = max(1, int(config.get("helper_max_retries", 3)))
        self.helper_retry_sleep_sec = max(0.0, float(config.get("helper_retry_sleep_sec", 0.5)))
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            mode = "local"
        except Exception:
            self.model = None
            mode = "external"
            if not self.external_python:
                raise RuntimeError("ultralytics unavailable locally and no external python_executable configured.")
            if not Path(self.external_python).exists():
                raise FileNotFoundError(f"Configured external python_executable not found: {self.external_python}")
            if not Path(self.helper_script).exists():
                raise FileNotFoundError(f"YOLO-World helper script not found: {self.helper_script}")

        self.last_generation_info = {
            "backend": "yoloworld",
            "model_path": self.model_path,
            "device": self.device,
            "class_count": len(self.classes),
            "mode": mode,
        }

    def generate_anchors(self, rgb: np.ndarray) -> List[Anchor2D]:
        if self.model is None and not self.external_python:
            raise RuntimeError("YOLOWorldAnchorBackend.initialize() must be called before inference.")

        anchors: list[Anchor2D] = []
        if self.model is not None:
            if self.classes:
                try:
                    self.model.set_classes(self.classes)
                except Exception:
                    logger.debug("YOLO-World set_classes failed; continuing with model defaults.", exc_info=True)

            results = self.model.predict(
                source=rgb,
                conf=self.confidence_threshold,
                verbose=False,
                max_det=self.max_detections,
                device=self.device,
            )

            if results:
                result = results[0]
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {})
                if boxes is not None:
                    for anchor_id, (xyxy, conf, cls_idx) in enumerate(
                        zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
                    ):
                        class_name = names.get(int(cls_idx), str(int(cls_idx))) if isinstance(names, dict) else str(int(cls_idx))
                        anchors.append(
                            Anchor2D(
                                anchor_id=int(anchor_id),
                                bbox_xyxy=np.asarray(xyxy, dtype=np.float32),
                                class_name=str(class_name),
                                confidence=float(conf),
                                metadata={"source": "yoloworld"},
                            )
                        )
        else:
            anchors = self._generate_anchors_via_helper(rgb)

        self.last_generation_info = {
            **self.last_generation_info,
            "anchor_count": int(len(anchors)),
        }
        return anchors

    def _generate_anchors_via_helper(self, rgb: np.ndarray) -> List[Anchor2D]:
        last_error: RuntimeError | None = None
        for attempt in range(1, self.helper_max_retries + 1):
            with tempfile.TemporaryDirectory(prefix="oviovo_yoloworld_") as tmpdir:
                image_path = Path(tmpdir) / "frame.png"
                output_json = Path(tmpdir) / "anchors.json"
                try:
                    # Match the old cv2.imwrite(rgb) behavior so helper-side cv2.imread
                    # sees the same channel order as before.
                    Image.fromarray(np.asarray(rgb)[..., ::-1]).save(image_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to write temporary anchor image: {image_path}") from exc
                cmd = [
                    self.external_python,
                    self.helper_script,
                    "--image-path",
                    str(image_path),
                    "--model-path",
                    self.model_path,
                    "--output-json",
                    str(output_json),
                    "--classes-json",
                    json.dumps(self.classes),
                    "--confidence-threshold",
                    str(self.confidence_threshold),
                    "--max-detections",
                    str(self.max_detections),
                    "--device",
                    self.device,
                ]
                env = dict(os.environ)
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("OPENBLAS_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                env.setdefault("NUMEXPR_NUM_THREADS", "1")
                env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
                env.setdefault("YOLO_CONFIG_DIR", str(Path(tmpdir) / "ultralytics"))
                env.setdefault("MPLCONFIGDIR", str(Path(tmpdir) / "mplconfig"))
                env.setdefault("XDG_CACHE_HOME", str(Path(tmpdir) / ".cache"))
                if str(self.device).strip().lower().startswith("cpu"):
                    env["CUDA_VISIBLE_DEVICES"] = ""
                completed = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                if completed.returncode == 0 and output_json.exists():
                    payload = json.loads(output_json.read_text(encoding="utf-8"))
                    break
                last_error = RuntimeError(
                    "YOLO-World helper failed "
                    f"(attempt={attempt}/{self.helper_max_retries} exit={completed.returncode}) "
                    f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
            if attempt < self.helper_max_retries:
                time.sleep(self.helper_retry_sleep_sec)
        else:
            assert last_error is not None
            raise last_error
        anchors = []
        for item in payload.get("anchors", []):
            anchors.append(
                Anchor2D(
                    anchor_id=int(item["anchor_id"]),
                    bbox_xyxy=np.asarray(item["bbox_xyxy"], dtype=np.float32),
                    class_name=str(item["class_name"]),
                    confidence=float(item["confidence"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return anchors
