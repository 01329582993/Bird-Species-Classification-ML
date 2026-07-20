"""Utilities for dataset preprocessing."""

from pathlib import Path


def list_dataset_files(dataset_dir: str) -> list[Path]:
    """Return a sorted list of files from the dataset directory."""
    return sorted(Path(dataset_dir).glob("**/*"))
