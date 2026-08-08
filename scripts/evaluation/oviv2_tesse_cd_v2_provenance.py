"""Formal environment and repository provenance for the TESSE-CD v2 protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def collect_formal_environment() -> dict[str, Any]:
    try:
        gpu = [
            line.strip()
            for line in subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    try:
        nvcc = [
            line.strip()
            for line in subprocess.run(
                ["nvcc", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError):
        nvcc = []
    try:
        import torch

        torch_cuda = torch.version.cuda or "unavailable"
        cudnn = torch.backends.cudnn.version()
    except (AttributeError, ImportError, OSError, RuntimeError):
        torch_cuda = "unavailable"
        cudnn = None
    libraries: dict[str, str] = {}
    for distribution in ("numpy", "open3d", "scipy", "torch", "pillow"):
        try:
            libraries[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            libraries[distribution] = "unavailable"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        visible = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform() or "unknown",
        "machine": platform.machine() or "unknown",
        "host": platform.node() or "unknown",
        "cuda": [
            f"torch_cuda={torch_cuda}",
            f"cudnn={cudnn if cudnn is not None else 'unavailable'}",
            *([f"nvcc={line}" for line in nvcc] if nvcc else ["nvcc=unavailable"]),
        ],
        "cuda_visible_devices": visible,
        "gpu": gpu or ["unavailable"],
        "libraries": libraries,
    }


def repository_provenance() -> dict[str, str]:
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip().decode("utf-8")
        if Path(top_level).resolve() != REPO_ROOT.resolve():
            raise ValueError(
                "repository root mismatch; formal execution requires REPO_ROOT "
                "to be the deployed Git worktree root"
            )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip().decode("ascii")
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip().decode("ascii")
        dirty_state = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise ValueError(
            "repository provenance unavailable; formal execution requires a "
            "deployed Git worktree with a resolvable commit and tree"
        ) from exc
    return {
        "repository_commit": commit,
        "repository_tree": tree,
        "dirty_state_digest": hashlib.sha256(dirty_state).hexdigest(),
    }
