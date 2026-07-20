"""Feature extraction helpers for bird image classification."""

from pathlib import Path
from typing import Iterable


def load_image_paths(image_dir: str) -> list[Path]:
    """Collect image file paths from a directory."""
    return sorted(Path(image_dir).glob("**/*"))


def describe_feature_set(features: Iterable[float]) -> dict[str, float]:
    """Return a simple summary of a feature iterable."""
    values = list(features)
    return {
        "count": float(len(values)),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }
