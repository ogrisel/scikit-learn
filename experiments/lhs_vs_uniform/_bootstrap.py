"""Prefer the installed scikit-learn over the unbuilt source tree in /workspace."""

from __future__ import annotations

import sys
from pathlib import Path


def prefer_installed_sklearn() -> None:
    workspace = Path(__file__).resolve().parents[2]
    cleaned = []
    for p in sys.path:
        if p in ("", "."):
            continue
        try:
            resolved = Path(p).resolve()
        except OSError:
            cleaned.append(p)
            continue
        # Drop the repo root so local ./sklearn does not shadow site-packages.
        if resolved == workspace:
            continue
        cleaned.append(p)
    sys.path[:] = cleaned


prefer_installed_sklearn()
