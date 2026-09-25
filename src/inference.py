"""Lightweight model-loading helpers for Kaggle notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import List


def find_model_files(root: str | Path) -> List[Path]:
    """Find common PyTorch checkpoint files under a root folder."""
    root = Path(root)
    patterns = ["*.pth", "*.pth.tar", "*.pt", "*.bin", "*.safetensors", "*.ckpt", "*.tar"]
    candidates = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    return sorted(set(candidates))


def select_checkpoint(candidates: List[Path], keyword: str | None = None) -> Path:
    """Select a checkpoint, optionally preferring paths containing a keyword."""
    if not candidates:
        raise FileNotFoundError("No checkpoint files found.")

    if keyword:
        preferred = [p for p in candidates if keyword.lower() in str(p).lower()]
        if preferred:
            return preferred[0]

    return candidates[0]
