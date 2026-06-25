"""YOLOE-Seg prompt-free 2D anchor backend."""

from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, List

import numpy as np
from PIL import Image

from src.core.data_structures import Anchor2D
from src.models.json_line_worker_client import JsonLineWorkerClient
from src.models.object_anchor_backend import ObjectAnchorBackend

logger = logging.getLogger("oviovo.models.yoloe_seg_anchor_backend")


class YOLOESegAnchorBackend(ObjectAnchorBackend):
    """Generate YOLOE-Seg anchors from prompt-free checkpoints, optionally with a vocabulary."""

    def __init__(self) -> None:
        self.external_python = ""
        self.helper_script = ""
        self.repo_root = ""
        self.checkpoint_path = ""
        self.vocab_source_checkpoint_path = ""
        self.device = "cuda:0"
        self.classes: list[str] = []
        self.confidence_threshold = 0.01
        self.max_detections = 128
        self.helper_max_retries = 3
        self.helper_retry_sleep_sec = 0.5
        self.worker_enabled = False
        self.worker_script = ""
        self.worker_request_timeout_sec = 120.0
        self.worker_fallback_on_error = True
        self._worker_client = None
        self.last_generation_info: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any]) -> None:
        self.repo_root = str(config.get("repo_root", "/home/phl/vv/yoloe_repo_probe")).strip()
        self.checkpoint_path = str(
            config.get("checkpoint_path", "/home/phl/vv/yoloe_repo_probe/pretrain/yoloe-v8s-seg-pf.pt")
        ).strip()
        if not self.repo_root or not Path(self.repo_root).exists():
            raise FileNotFoundError(f"YOLOE repo_root not found: {self.repo_root}")
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"YOLOE checkpoint not found: {self.checkpoint_path}")

        self.device = str(config.get("device", "cuda:0")).strip() or "cuda:0"
        self.external_python = str(config.get("python_executable", "")).strip()
        helper_default = Path(__file__).resolve().parents[2] / "frontend" / "yoloe_seg_prompt_free_helper.py"
        self.helper_script = str(config.get("helper_script", helper_default)).strip()
        worker_default = Path(__file__).resolve().parents[2] / "frontend" / "yoloe_seg_prompt_free_worker.py"
        self.worker_script = str(config.get("worker_script", worker_default)).strip()
        self.classes = [str(item).strip() for item in config.get("classes", []) if str(item).strip()]
        self.vocab_source_checkpoint_path = self._resolve_vocab_source_checkpoint_path(config)
        self.confidence_threshold = float(config.get("confidence_threshold", 0.01))
        self.max_detections = int(config.get("max_detections", 128))
        self.helper_max_retries = max(1, int(config.get("helper_max_retries", 3)))
        self.helper_retry_sleep_sec = max(0.0, float(config.get("helper_retry_sleep_sec", 0.5)))
        self.worker_enabled = bool(config.get("worker_enabled", False))
        self.worker_request_timeout_sec = max(1.0, float(config.get("worker_request_timeout_sec", 120.0)))
        self.worker_fallback_on_error = bool(config.get("worker_fallback_on_error", True))
        if not self.external_python:
            raise RuntimeError("YOLOE-Seg backend requires python_executable.")
        if not Path(self.external_python).exists():
            raise FileNotFoundError(f"Configured YOLOE python_executable not found: {self.external_python}")
        if not Path(self.helper_script).exists():
            raise FileNotFoundError(f"YOLOE helper script not found: {self.helper_script}")
        if self.worker_enabled and not Path(self.worker_script).exists():
            raise FileNotFoundError(f"YOLOE worker script not found: {self.worker_script}")

        self.last_generation_info = {
            "backend": "yoloe_seg_pf",
            "repo_root": self.repo_root,
            "checkpoint_path": self.checkpoint_path,
            "vocab_source_checkpoint_path": self.vocab_source_checkpoint_path,
            "device": self.device,
            "class_count": len(self.classes),
            "worker_enabled": self.worker_enabled,
            "worker_fallback_on_error": self.worker_fallback_on_error,
            "mode": "external",
        }

    def generate_anchors(self, rgb: np.ndarray) -> List[Anchor2D]:
        mode = "external"
        if self.worker_enabled:
            try:
                anchors = self._generate_anchors_via_worker(rgb)
                mode = "worker"
            except Exception:
                self._reset_worker_client()
                if not self.worker_fallback_on_error:
                    raise
                logger.exception("YOLOE-Seg worker failed; falling back to helper process.")
                anchors = self._generate_anchors_via_helper(rgb)
                mode = "worker_fallback"
        else:
            anchors = self._generate_anchors_via_helper(rgb)
        self.last_generation_info = {
            **self.last_generation_info,
            "anchor_count": int(len(anchors)),
            "worker_enabled": self.worker_enabled,
            "worker_fallback_on_error": self.worker_fallback_on_error,
            "mode": mode,
        }
        return anchors

    def _resolve_vocab_source_checkpoint_path(self, config: dict[str, Any]) -> str:
        explicit = str(config.get("vocab_source_checkpoint_path", "")).strip()
        if explicit:
            path = Path(explicit)
            if not path.exists():
                raise FileNotFoundError(f"YOLOE vocab_source_checkpoint_path not found: {path}")
            return str(path)
        if not self.classes:
            return ""

        checkpoint = Path(self.checkpoint_path)
        if checkpoint.stem.endswith("-pf"):
            candidate = checkpoint.with_name(f"{checkpoint.stem[:-3]}{checkpoint.suffix}")
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "YOLOE vocabulary-aware prompt-free inference requires a non-PF source checkpoint; "
            f"could not infer one from {self.checkpoint_path!r}."
        )

    def _generate_anchors_via_helper(self, rgb: np.ndarray) -> List[Anchor2D]:
        last_error: RuntimeError | None = None
        payload: dict[str, Any] | None = None
        for attempt in range(1, self.helper_max_retries + 1):
            with tempfile.TemporaryDirectory(prefix="oviovo_yoloe_") as tmpdir:
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
                    "--checkpoint-path",
                    self.checkpoint_path,
                    "--repo-root",
                    self.repo_root,
                    "--output-json",
                    str(output_json),
                    "--classes-json",
                    json.dumps(self.classes),
                    "--vocab-source-checkpoint-path",
                    self.vocab_source_checkpoint_path,
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
                completed = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=self.repo_root,
                )
                if completed.returncode == 0 and output_json.exists():
                    payload = json.loads(output_json.read_text(encoding="utf-8"))
                    break
                last_error = RuntimeError(
                    "YOLOE-Seg helper failed "
                    f"(attempt={attempt}/{self.helper_max_retries} exit={completed.returncode}) "
                    f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
            if attempt < self.helper_max_retries:
                time.sleep(self.helper_retry_sleep_sec)
        else:
            assert last_error is not None
            raise last_error

        assert payload is not None
        return self._anchors_from_payload(payload)

    def _worker_command(self) -> list[str]:
        return [
            self.external_python,
            self.worker_script,
            "--checkpoint-path",
            self.checkpoint_path,
            "--repo-root",
            self.repo_root,
            "--classes-json",
            json.dumps(self.classes),
            "--vocab-source-checkpoint-path",
            self.vocab_source_checkpoint_path,
            "--confidence-threshold",
            str(self.confidence_threshold),
            "--max-detections",
            str(self.max_detections),
            "--device",
            self.device,
        ]

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        cache_root = Path(self.repo_root) / ".cache" / "oviovo_yoloe_worker"
        cache_root.mkdir(parents=True, exist_ok=True)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
        env.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
        env.setdefault("XDG_CACHE_HOME", str(cache_root / ".cache"))
        return env

    def _create_worker_client(self) -> JsonLineWorkerClient:
        return JsonLineWorkerClient(
            command=self._worker_command(),
            env=self._worker_env(),
            cwd=self.repo_root,
            request_timeout_sec=self.worker_request_timeout_sec,
        )

    def _encode_rgb_png_b64(self, rgb: np.ndarray) -> str:
        buffer = BytesIO()
        Image.fromarray(np.asarray(rgb)[..., ::-1]).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _anchors_from_payload(self, payload: dict[str, Any]) -> list[Anchor2D]:
        if payload.get("error"):
            raise RuntimeError(f"YOLOE-Seg worker failed: {payload['error']}")
        anchors: list[Anchor2D] = []
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

    def _generate_anchors_via_worker(self, rgb: np.ndarray) -> List[Anchor2D]:
        if self._worker_client is None:
            self._worker_client = self._create_worker_client()
        payload = self._worker_client.request({"image_png_b64": self._encode_rgb_png_b64(rgb)})
        return self._anchors_from_payload(payload)

    def _reset_worker_client(self) -> None:
        worker_client = self._worker_client
        self._worker_client = None
        if worker_client is None:
            return
        close = getattr(worker_client, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        self._reset_worker_client()
